
import argparse
import re
from pathlib import Path

import pandas as pd

from src.evaluate import prepare_benchmark


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def repository_of(issue_id):
    """Extract GitHub repository from a standardized issue ID."""
    if "#" not in str(issue_id):
        return None
    return str(issue_id).rsplit("#", 1)[0].lower()


def mentions_duplicate(text, target_id, dataset):
    """
    Check whether query text explicitly references a
    known earlier direct duplicate by its issue number.

    This is a conservative heuristic, not a complete
    detection of all references.
    """
    text = str(text)

    if dataset == "issuedupes":
        number = str(target_id).rsplit("#", 1)[-1]
    else:
        number = str(target_id)

    if not number.isdigit():
        return False

    pattern = (
        rf"(?<!\w)"
        rf"(?:bug|issue|ticket|report)"
        rf"\s*(?:#|id\s*[:#]?\s*|no\.?\s*|number\s*)?"
        rf"{re.escape(number)}(?!\d)"
        rf"|(?<!\w)#{re.escape(number)}(?!\d)"
    )

    return re.search(pattern, text, re.IGNORECASE) is not None


def audit_explicit_references(dataset):
    issues, queries, eligible, relevant = prepare_benchmark(
        dataset,
        "validation",
    )

    texts = issues.set_index("issue_id")["text"].to_dict()

    records = []

    for query in eligible.itertuples(index=False):
        query_id = str(query.issue_id)
        query_text = texts[query_id]

        earlier_duplicates = sorted(relevant[query_id])

        matched = [
            target
            for target in earlier_duplicates
            if mentions_duplicate(
                query_text, target, dataset
            )
        ]

        records.append({
            "query_id": query_id,
            "known_earlier_duplicates": len(
                earlier_duplicates
            ),
            "explicit_reference": bool(matched),
            "referenced_duplicate_ids": ", ".join(
                matched
            ),
        })

    frame = pd.DataFrame(records)

    count = int(frame["explicit_reference"].sum())

    print("\nExplicit duplicate-reference audit")
    print("-" * 50)
    print("Dataset:", dataset)
    print("Eligible validation queries:", len(frame))
    print("Queries with explicit references:", count)
    print(
        "Reference rate:",
        f"{count / len(frame):.2%}",
    )

    if count:
        print("\nExample matches:")
        print(
            frame.loc[
                frame["explicit_reference"],
                ["query_id", "referenced_duplicate_ids"],
            ].head(10).to_string(index=False)
        )

    return frame


def audit_repository_interference(dataset):
    path = (
        RESULTS
        / f"{dataset}_hybrid_candidate_inspection.csv"
    )

    if not path.exists():
        print("\nNo candidate-inspection CSV:", path)
        return None

    frame = pd.read_csv(
        path,
        dtype={
            "query_id": str,
            "candidate_id": str,
        },
    )

    required = {
        "query_id", "candidate_id",
        "method", "rank", "direct_duplicate",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing inspection columns: {sorted(missing)}"
        )

    frame = frame[frame["rank"] == 1].copy()

    # Only examine top-ranked candidates from
    # the queries we explicitly inspected.
    frame["query_repo"] = frame["query_id"].map(
        repository_of
    )

    frame["candidate_repo"] = frame[
        "candidate_id"
    ].map(repository_of)

    frame["cross_repository"] = (
        frame["query_repo"].notna()
        & frame["candidate_repo"].notna()
        & (
            frame["query_repo"]
            != frame["candidate_repo"]
        )
    )

    print("\nRepository interference audit")
    print("-" * 50)
    print("Dataset:", dataset)

    if dataset != "issuedupes":
        print(
            "Repository comparison is not applicable "
            "to Eclipse issue IDs."
        )
        return None

    print(
        "Inspected queries:",
        frame["query_id"].nunique(),
    )

    summary = (
        frame.groupby("method")
        .agg(
            queries=("query_id", "nunique"),
            cross_repo_top1=("cross_repository", "sum"),
        )
        .reset_index()
    )

    summary["cross_repo_rate"] = (
        summary["cross_repo_top1"]
        / summary["queries"]
    )

    print(summary.to_string(index=False))

    print(
        "\nNote: These results concern only the "
        "manually selected inspection queries."
    )

    return frame


def main(args):
    RESULTS.mkdir(parents=True, exist_ok=True)

    references = audit_explicit_references(
        args.dataset
    )

    references.to_csv(
        RESULTS
        / f"{args.dataset}_explicit_reference_audit.csv",
        index=False,
    )

    repository_results = audit_repository_interference(
        args.dataset
    )

    if repository_results is not None:
        repository_results.to_csv(
            RESULTS
            / f"{args.dataset}_repository_inspection_audit.csv",
            index=False,
        )

    print("\nAudit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    main(parser.parse_args())

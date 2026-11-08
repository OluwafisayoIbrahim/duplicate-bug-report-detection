
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"


def preview(text, limit=650):
    text = str(text).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def inspect_report(issue_id, issues, label):
    print(f"\n{label}: {issue_id}")

    if issue_id not in issues.index:
        print("Report not found.")
        return

    report = issues.loc[issue_id]

    print("Group:", report["group_id"])
    print("Characters:", len(str(report["text"])))
    print("Text:", preview(report["text"]))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        default="eclipse",
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    model = args.model or f"{args.dataset}-pilot"

    comparison_path = (
        RESULTS
        / f"{args.dataset}_{model}_test_error_comparison.csv"
    )

    comparison = pd.read_csv(
        comparison_path,
        dtype={"query_id": str},
    )

    issues = pd.read_csv(
        PROCESSED / f"{args.dataset}_issues_grouped.csv",
        dtype={"issue_id": str, "group_id": str},
    )

    issues = issues.drop_duplicates("issue_id")
    issues = issues.set_index("issue_id")

    cases = [
        (
            "LARGEST IMPROVEMENTS",
            comparison.nlargest(args.limit, "rr_change"),
        ),
        (
            "LARGEST REGRESSIONS",
            comparison.nsmallest(args.limit, "rr_change"),
        ),
    ]

    for heading, selected in cases:
        print("\n" + "=" * 75)
        print(heading)
        print("=" * 75)

        for row in selected.itertuples(index=False):
            query_id = str(row.query_id)

            pretrained_id = str(
                row.top_candidate_id_pretrained
            )

            finetuned_id = str(
                row.top_candidate_id_finetuned
            )

            print("\n" + "-" * 75)
            print("QUERY:", query_id)

            print(
                "First relevant rank:",
                row.first_relevant_rank_pretrained,
                "->",
                row.first_relevant_rank_finetuned,
            )

            for label, issue_id in [
                ("QUERY REPORT", query_id),
                ("PRETRAINED TOP CANDIDATE", pretrained_id),
                ("FINE-TUNED TOP CANDIDATE", finetuned_id),
            ]:
                inspect_report(issue_id, issues, label)

            if all(
                issue_id in issues.index
                for issue_id in (
                    query_id,
                    pretrained_id,
                    finetuned_id,
                )
            ):
                query_group = issues.loc[query_id, "group_id"]

                print("\nKnown group membership")
                print(
                    "Pretrained same group:",
                    issues.loc[pretrained_id, "group_id"]
                    == query_group,
                )
                print(
                    "Fine-tuned same group:",
                    issues.loc[finetuned_id, "group_id"]
                    == query_group,
                )


if __name__ == "__main__":
    main()

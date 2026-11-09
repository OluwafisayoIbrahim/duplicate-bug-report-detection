
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def verify(dataset):
    issues = pd.read_csv(
        PROCESSED / f"{dataset}_issues_grouped.csv",
        dtype={"issue_id": str},
    )

    pairs = pd.read_csv(
        PROCESSED / f"{dataset}_fit_pairs.csv",
        dtype={"source_id": str, "target_id": str},
    )

    issues["created_at"] = pd.to_datetime(
        issues["created_at"],
        utc=True,
        errors="coerce",
    )

    issues["text"] = issues["text"].fillna("").astype(str)

    issues = issues[
        issues["created_at"].notna()
        & issues["text"].str.strip().ne("")
    ].copy()

    issues = issues.sort_values(
        ["created_at", "issue_id"]
    ).reset_index(drop=True)

    outer_idx = int(len(issues) * 0.8)
    train = issues.iloc[:outer_idx]

    inner_idx = int(len(train) * 0.85)
    fitting = train.iloc[:inner_idx]
    validation = train.iloc[inner_idx:]
    test = issues.iloc[outer_idx:]

    fitting_ids = set(fitting["issue_id"])
    validation_ids = set(validation["issue_id"])
    test_ids = set(test["issue_id"])

    pair_ids = set(pairs["source_id"]) | set(
        pairs["target_id"]
    )

    outside_fitting = pair_ids - fitting_ids

    print(f"\n{'=' * 55}")
    print("Dataset:", dataset)
    print("=" * 55)

    print("Saved training pairs:", len(pairs))
    print("Unique issues in pairs:", len(pair_ids))

    print(
        "Pair endpoints outside fitting period:",
        len(outside_fitting),
    )

    print(
        "Pair endpoints in validation:",
        len(pair_ids & validation_ids),
    )

    print(
        "Pair endpoints in test:",
        len(pair_ids & test_ids),
    )

    missing_text = (
        pairs["text_a"].isna()
        | pairs["text_b"].isna()
        | pairs["text_a"].astype(str).str.strip().eq("")
        | pairs["text_b"].astype(str).str.strip().eq("")
    ).sum()

    self_pairs = (
        pairs["source_id"] == pairs["target_id"]
    ).sum()

    print("Empty-text pairs:", int(missing_text))
    print("Self-referencing pairs:", int(self_pairs))

    if outside_fitting:
        print(
            "Example unexpected IDs:",
            sorted(outside_fitting)[:10],
        )

    passed = (
        len(outside_fitting) == 0
        and missing_text == 0
        and self_pairs == 0
    )

    print("Verification passed:", passed)


for dataset in ("issuedupes", "eclipse"):
    verify(dataset)

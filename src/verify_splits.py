
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

DATASETS = ["issuedupes", "eclipse"]


def verify_dataset(dataset):
    issues = pd.read_csv(
        PROCESSED / f"{dataset}_issues_grouped.csv",
        dtype={"issue_id": str},
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

    notebook_train = issues.iloc[:outer_idx]
    notebook_test = issues.iloc[outer_idx:]

    inner_idx = int(len(notebook_train) * 0.85)

    notebook_fit = notebook_train.iloc[:inner_idx]
    notebook_validation = notebook_train.iloc[inner_idx:]

    outer_cutoff = issues.iloc[outer_idx]["created_at"]

    eval_train = issues[
        issues["created_at"] < outer_cutoff
    ].copy()

    eval_test = issues[
        issues["created_at"] >= outer_cutoff
    ].copy()

    inner_eval_idx = int(len(eval_train) * 0.85)
    inner_cutoff = eval_train.iloc[
        inner_eval_idx
    ]["created_at"]

    eval_fit = eval_train[
        eval_train["created_at"] < inner_cutoff
    ]

    eval_validation = eval_train[
        eval_train["created_at"] >= inner_cutoff
    ]

    print(f"\n{'=' * 55}")
    print("Dataset:", dataset)
    print("=" * 55)

    partitions = {
        "Training period": (
            notebook_train,
            eval_train,
        ),
        "Test": (
            notebook_test,
            eval_test,
        ),
        "Fitting period": (
            notebook_fit,
            eval_fit,
        ),
        "Validation": (
            notebook_validation,
            eval_validation,
        ),
    }

    for name, (notebook, evaluation) in partitions.items():
        notebook_ids = set(notebook["issue_id"])
        evaluation_ids = set(evaluation["issue_id"])

        print(f"\n{name}")
        print("Notebook:", len(notebook_ids))
        print("Evaluation:", len(evaluation_ids))
        print(
            "Membership matches:",
            notebook_ids == evaluation_ids,
        )
        print(
            "Symmetric difference:",
            len(notebook_ids ^ evaluation_ids),
        )

    print("\nCutoff timestamps")
    print("Outer:", outer_cutoff)
    print("Inner:", inner_cutoff)


for dataset in DATASETS:
    verify_dataset(dataset)

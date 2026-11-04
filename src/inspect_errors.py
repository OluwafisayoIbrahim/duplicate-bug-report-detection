
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

comparison = pd.read_csv(
    ROOT / "results"
    / "eclipse_eclipse-pilot_test_error_comparison.csv",
    dtype={"query_id": str},
)

issues = pd.read_csv(
    ROOT / "data/processed/eclipse_issues_grouped.csv",
    dtype={"issue_id": str},
)

issues = issues.drop_duplicates("issue_id").set_index("issue_id")

def inspect_cases(cases, heading):
    print(f"\n{'=' * 65}")
    print(heading)
    print("=" * 65)

    for row in cases.itertuples(index=False):
        issue_id = row.query_id

        if issue_id not in issues.index:
            print(f"\nIssue {issue_id}: not found")
            continue

        issue = issues.loc[issue_id]
        report_text = str(issue["text"])

        print(f"\nIssue ID: {issue_id}")
        print(
            "Rank:",
            row.first_relevant_rank_pretrained,
            "->",
            row.first_relevant_rank_finetuned,
        )
        print("Text length (characters):", len(report_text))
        print("Group:", issue["group_id"])
        print("Report excerpt:")
        print(report_text[:1000])
        print("-" * 65)


inspect_cases(
    comparison.nlargest(5, "rr_change"),
    "FIVE LARGEST IMPROVEMENTS",
)

inspect_cases(
    comparison.nsmallest(5, "rr_change"),
    "FIVE LARGEST REGRESSIONS",
)

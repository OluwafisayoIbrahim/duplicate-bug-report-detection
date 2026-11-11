
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

METRICS = {
    "Hit@1": "hit_at_1",
    "Hit@5": "hit_at_5",
    "Hit@10": "hit_at_10",
    "MRR": "reciprocal_rank",
}


def load_results(path):
    frame = pd.read_csv(
        path,
        dtype={"query_id": str},
    )

    if not frame["query_id"].is_unique:
        raise ValueError(
            f"Duplicate query IDs in {path}"
        )

    return frame


def main(args):
    prefix = (
        f"{args.dataset}_{args.model}"
        "_fusion_validation"
    )

    baseline = load_results(
        RESULTS / f"{prefix}_minilm_queries.csv"
    )

    hybrid = load_results(
        RESULTS / f"{prefix}_full_w95_queries.csv"
    )

    audit = load_results(
        RESULTS
        / f"{args.dataset}_explicit_reference_audit.csv"
    )

    combined = baseline.merge(
        hybrid,
        on="query_id",
        suffixes=("_minilm", "_hybrid"),
        validate="one_to_one",
    )

    combined = combined.merge(
        audit[["query_id", "explicit_reference"]],
        on="query_id",
        validate="one_to_one",
    )

    if not (
        len(combined)
        == len(baseline)
        == len(hybrid)
        == len(audit)
    ):
        raise ValueError("Query sets do not match.")

    # Normalize the saved reference flag.
    combined["explicit_reference"] = (
        combined["explicit_reference"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if combined["explicit_reference"].isna().any():
        raise ValueError(
            "Invalid explicit_reference values."
        )

    subsets = {
        "All queries": combined,
        "Without explicit references": combined[
            ~combined["explicit_reference"]
        ],
        "Explicit references only": combined[
            combined["explicit_reference"]
        ],
    }

    records = []

    print("\nReference sensitivity analysis")
    print("=" * 55)
    print("Dataset:", args.dataset)

    for label, subset in subsets.items():
        print(f"\n{label}: {len(subset)} queries")

        for metric, column in METRICS.items():
            baseline_mean = subset[
                f"{column}_minilm"
            ].mean()

            hybrid_mean = subset[
                f"{column}_hybrid"
            ].mean()

            delta = hybrid_mean - baseline_mean

            records.append({
                "subset": label,
                "queries": len(subset),
                "metric": metric,
                "minilm": baseline_mean,
                "hybrid": hybrid_mean,
                "delta": delta,
            })

            print(
                f"{metric}: "
                f"MiniLM={baseline_mean:.4f}, "
                f"Hybrid={hybrid_mean:.4f}, "
                f"Delta={delta:+.5f}"
            )

    output = (
        RESULTS
        / f"{args.dataset}_reference_sensitivity.csv"
    )

    pd.DataFrame(records).to_csv(
        output, index=False
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=[
            "issuedupes-pilot",
            "eclipse-pilot",
        ],
        required=True,
    )

    main(parser.parse_args())

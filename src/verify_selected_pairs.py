
from pathlib import Path

import pandas as pd

from src.train import select_group_safe_pairs


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def canonical_pair(a, b):
    return tuple(sorted((str(a), str(b))))


def verify(dataset):
    pairs = pd.read_csv(
        PROCESSED / f"{dataset}_fit_pairs.csv",
        dtype={"source_id": str, "target_id": str},
    )

    selected = select_group_safe_pairs(pairs)

    canonical = [
        canonical_pair(a, b)
        for a, b in selected[
            ["source_id", "target_id"]
        ].itertuples(index=False, name=None)
    ]

    duplicate_count = len(canonical) - len(set(canonical))

    repeated_groups = selected["train_group"].duplicated().sum()

    print(f"\n{'=' * 55}")
    print("Dataset:", dataset)
    print("=" * 55)

    print("Original pairs:", len(pairs))
    print("Selected training examples:", len(selected))
    print("Repeated unordered pairs:", duplicate_count)
    print("Repeated training groups:", int(repeated_groups))
    print("Verification passed:", (
        duplicate_count == 0 and repeated_groups == 0
    ))


for dataset in ("issuedupes", "eclipse"):
    verify(dataset)

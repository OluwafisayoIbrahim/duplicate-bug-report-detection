
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def canonical_pair(a, b):
    return tuple(sorted((str(a), str(b))))


def verify(dataset):
    pairs = pd.read_csv(
        PROCESSED / f"{dataset}_fit_pairs.csv",
        dtype={"source_id": str, "target_id": str},
    )

    links = pd.read_csv(
        PROCESSED / f"{dataset}_direct_links.csv",
        dtype={"source_id": str, "target_id": str},
    )

    known_links = {
        canonical_pair(source, target)
        for source, target in links[
            ["source_id", "target_id"]
        ].itertuples(index=False, name=None)
        if source != target
    }

    training_links = [
        canonical_pair(source, target)
        for source, target in pairs[
            ["source_id", "target_id"]
        ].itertuples(index=False, name=None)
    ]

    unmatched = [
        pair for pair in training_links
        if pair not in known_links
    ]

    duplicate_pairs = (
        len(training_links) - len(set(training_links))
    )

    print(f"\n{'=' * 55}")
    print("Dataset:", dataset)
    print("=" * 55)

    print("Saved training pairs:", len(training_links))
    print("Unique training pairs:", len(set(training_links)))
    print("Pairs without recorded direct links:", len(unmatched))
    print("Repeated unordered training pairs:", duplicate_pairs)

    if unmatched:
        print("Examples:", unmatched[:10])

    passed = not unmatched and duplicate_pairs == 0

    print("Verification passed:", passed)


for dataset in ("issuedupes", "eclipse"):
    verify(dataset)

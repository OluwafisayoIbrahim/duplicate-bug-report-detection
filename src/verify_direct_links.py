
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"

links = pd.read_csv(
    PROCESSED / "eclipse_direct_links.csv",
    dtype={"source_id": str, "target_id": str},
)

comparison = pd.read_csv(
    RESULTS / "eclipse_eclipse-pilot_test_error_comparison.csv",
    dtype={
        "query_id": str,
        "top_candidate_id_pretrained": str,
        "top_candidate_id_finetuned": str,
    },
)

# Direct relationships are treated as undirected
# for our retrieval evaluation.
adjacency = {}

for source, target in links[
    ["source_id", "target_id"]
].itertuples(index=False, name=None):
    adjacency.setdefault(source, set()).add(target)
    adjacency.setdefault(target, set()).add(source)


def is_direct_duplicate(query_id, candidate_id):
    return candidate_id in adjacency.get(query_id, set())


selected = pd.concat(
    [
        comparison.nlargest(3, "rr_change"),
        comparison.nsmallest(3, "rr_change"),
    ],
    ignore_index=True,
)

print("\nDirect duplicate verification")
print("=" * 60)

for row in selected.itertuples(index=False):
    query_id = row.query_id

    pretrained_id = row.top_candidate_id_pretrained
    finetuned_id = row.top_candidate_id_finetuned

    print(f"\nQuery: {query_id}")
    print(
        "Rank:",
        row.first_relevant_rank_pretrained,
        "->",
        row.first_relevant_rank_finetuned,
    )

    print(
        f"Pretrained candidate ({pretrained_id}):",
        is_direct_duplicate(query_id, pretrained_id),
    )

    print(
        f"Fine-tuned candidate ({finetuned_id}):",
        is_direct_duplicate(query_id, finetuned_id),
    )

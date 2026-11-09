
import numpy as np
import pandas as pd

from src.evaluate import evaluate_retrieval


issues = pd.DataFrame({
    "issue_id": ["A", "B", "C"],
    "created_at": pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        utc=True,
    ),
})

eligible = issues.iloc[[2]].copy()

relevant = {
    "C": {"A"},
}

embeddings = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
], dtype=np.float32)

results, queries = evaluate_retrieval(
    issues,
    eligible,
    relevant,
    embeddings,
)

assert results["Queries"] == 1
assert results["Hit@1"] == 1.0
assert results["MRR"] == 1.0
assert queries.iloc[0]["top_candidate_id"] == "A"

print("Ranking regression test: PASS")

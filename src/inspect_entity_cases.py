
import json
from pathlib import Path

import pandas as pd

from src.evaluate import prepare_benchmark


ROOT = Path(__file__).resolve().parents[1]

QUERY_IDS = [
    "175595",
    "161930",
    "174759",
    "172575",
]


def main():
    issues, queries, eligible, relevant = prepare_benchmark(
        "eclipse",
        "validation",
    )

    with (
        ROOT / "models/entity_cache/eclipse_entities_v1.json"
    ).open("r", encoding="utf-8") as f:
        cached = json.load(f)

    if len(cached["entities"]) != len(issues):
        raise ValueError("Entity cache length mismatch.")

    import hashlib

    digest = hashlib.sha256()

    for issue_id, text in zip(
        issues["issue_id"].astype(str),
        issues["text"].fillna(""),
    ):
        digest.update(issue_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")

    if cached["signature"] != digest.hexdigest():
        raise ValueError("Entity cache signature mismatch.")

    id_to_idx = {
        str(issue_id): idx
        for idx, issue_id in enumerate(issues["issue_id"])
    }

    entities = [
        set(items)
        for items in cached["entities"]
    ]

    for query_id in QUERY_IDS:
        if query_id not in id_to_idx:
            print(f"Missing query: {query_id}")
            continue

        query_idx = id_to_idx[query_id]
        query_entities = entities[query_idx]

        print("\n" + "=" * 70)
        print("QUERY:", query_id)
        print("=" * 70)

        print(
            "Query text:",
            " ".join(
                str(issues.iloc[query_idx]["text"]).split()
            )[:350],
        )

        print("\nQuery entities:")
        print(sorted(query_entities)[:40])

        for duplicate_id in sorted(relevant[query_id]):
            duplicate_id = str(duplicate_id)

            if duplicate_id not in id_to_idx:
                continue

            dup_idx = id_to_idx[duplicate_id]
            duplicate_entities = entities[dup_idx]

            shared = query_entities & duplicate_entities

            union = query_entities | duplicate_entities

            similarity = (
                len(shared) / len(union)
                if union else 0.0
            )

            print("\nKnown duplicate:", duplicate_id)
            print("Jaccard:", round(similarity, 4))
            print("Shared entities:", sorted(shared)[:50])

            print(
                "Duplicate text:",
                " ".join(
                    str(issues.iloc[dup_idx]["text"]).split()
                )[:350],
            )


if __name__ == "__main__":
    main()

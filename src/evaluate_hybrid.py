
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from src.evaluate import prepare_benchmark, load_model


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def get_ranks(scores):
    """Return 1-based ranks with stable tie ordering."""
    order = np.argsort(-scores, kind="stable")

    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)

    return ranks


def evaluate_hybrid(
    issues,
    eligible,
    relevant,
    tfidf_matrix,
    embeddings,
    rrf_k=60,
):
    issue_ids = issues["issue_id"].to_numpy()

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    id_to_idx = {
        issue_id: i
        for i, issue_id in enumerate(issue_ids)
    }

    records = []

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc="Hybrid retrieval",
    ):
        query_idx = id_to_idx[query.issue_id]

        cutoff = np.searchsorted(
            timestamps,
            np.datetime64(query.created_at.to_datetime64()),
            side="left",
        )

        if cutoff == 0:
            continue

        tfidf_scores = (
            tfidf_matrix[:cutoff]
            @ tfidf_matrix[query_idx].T
        ).toarray().ravel()

        semantic_scores = np.asarray(
            embeddings[:cutoff] @ embeddings[query_idx]
        )

        lexical_ranks = get_ranks(tfidf_scores)
        semantic_ranks = get_ranks(semantic_scores)

        fused_scores = (
            1.0 / (rrf_k + lexical_ranks)
            + 1.0 / (rrf_k + semantic_ranks)
        )

        fused_ranks = get_ranks(fused_scores)

        relevant_indices = [
            id_to_idx[rid]
            for rid in relevant[query.issue_id]
            if rid in id_to_idx and id_to_idx[rid] < cutoff
        ]

        if not relevant_indices:
            raise ValueError(
                f"No earlier duplicate for {query.issue_id}"
            )

        best_rank = int(
            min(fused_ranks[relevant_indices])
        )

        top_idx = int(np.argmin(fused_ranks))

        records.append({
            "query_id": query.issue_id,
            "first_relevant_rank": best_rank,
            "hit_at_1": int(best_rank <= 1),
            "hit_at_5": int(best_rank <= 5),
            "hit_at_10": int(best_rank <= 10),
            "reciprocal_rank": 1.0 / best_rank,
            "top_candidate_id": issue_ids[top_idx],
        })

    if not records:
        raise ValueError("No queries evaluated.")

    results = pd.DataFrame(records)

    metrics = {
        "Queries": len(results),
        "Hit@1": results["hit_at_1"].mean(),
        "Hit@5": results["hit_at_5"].mean(),
        "Hit@10": results["hit_at_10"].mean(),
        "MRR": results["reciprocal_rank"].mean(),
    }

    return metrics, results


def main(args):
    if args.split != "validation":
        raise ValueError(
            "Hybrid development must start on validation."
        )

    issues, queries, eligible, relevant = prepare_benchmark(
        args.dataset,
        args.split,
    )

    print("Dataset:", args.dataset)
    print("Model:", args.model)
    print("Eligible queries:", len(eligible))

    first_validation_time = queries["created_at"].min()

    fit_issues = issues[
        issues["created_at"] < first_validation_time
    ]

    print("TF-IDF fitting reports:", len(fit_issues))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )

    vectorizer.fit(fit_issues["text"].tolist())

    tfidf_matrix = vectorizer.transform(
        issues["text"].tolist()
    )

    print("TF-IDF matrix:", tfidf_matrix.shape)

    model = load_model(args.model)

    print("\nGenerating MiniLM embeddings...")

    embeddings = model.encode(
        issues["text"].tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("Embedding shape:", embeddings.shape)

    metrics, query_results = evaluate_hybrid(
        issues,
        eligible,
        relevant,
        tfidf_matrix,
        embeddings,
        rrf_k=args.rrf_k,
    )

    print("\nHybrid retrieval results")
    print("-" * 40)

    for name, value in metrics.items():
        if name == "Queries":
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    prefix = (
        f"{args.dataset}_{args.model}"
        f"_rrf{args.rrf_k}_{args.split}"
    )

    pd.DataFrame([{
        "Dataset": args.dataset,
        "Model": args.model,
        "Method": "TF-IDF + MiniLM RRF",
        "Split": args.split,
        "RRF_k": args.rrf_k,
        **metrics,
    }]).to_csv(
        RESULTS / f"{prefix}.csv",
        index=False,
    )

    query_results.to_csv(
        RESULTS / f"{prefix}_queries.csv",
        index=False,
    )

    print("\nResults saved:", prefix)


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
            "pretrained",
            "issuedupes-pilot",
            "eclipse-pilot",
        ],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["validation"],
        default="validation",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    main(parser.parse_args())


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from src.evaluate import (
    MODEL_PATHS,
    load_model,
    prepare_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / "models" / "hybrid_cache"

RRF_K = 60
WEIGHTS = [0.50, 0.70, 0.85, 0.95]
TOP_NS = [0, 50, 200]


def fingerprint(issues, model_name):
    h = hashlib.sha256()

    h.update(model_name.encode())
    h.update(b"max_seq_length=256;normalized=True")

    for row in issues[
        ["issue_id", "created_at", "text"]
    ].itertuples(index=False, name=None):
        for value in row:
            h.update(str(value).encode("utf-8"))
            h.update(b"\x00")

    model_path = MODEL_PATHS[model_name]

    if isinstance(model_path, Path):
        for path in sorted(model_path.rglob("*")):
            if path.is_file():
                stat = path.stat()
                h.update(str(path.relative_to(model_path)).encode())
                h.update(str(stat.st_size).encode())
                h.update(str(stat.st_mtime_ns).encode())
    else:
        h.update(str(model_path).encode())

    return h.hexdigest()[:20]


def load_or_create_embeddings(
    issues, model_name, dataset, split, batch_size
):
    CACHE.mkdir(parents=True, exist_ok=True)

    signature = fingerprint(issues, model_name)

    path = (
        CACHE
        / f"{dataset}_{model_name}_{split}_{signature}.npy"
    )

    if path.exists():
        print("Loading cached MiniLM embeddings:", path)

        embeddings = np.load(path, allow_pickle=False)

        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(issues)
            or embeddings.shape[1] != 384
        ):
            raise ValueError("Cached embedding shape mismatch.")

        return embeddings

    print("\nGenerating MiniLM embeddings...")

    model = load_model(model_name)

    embeddings = model.encode(
        issues["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(np.float32)

    np.save(path, embeddings)

    print("Embeddings cached:", path)

    return embeddings


def get_ranks(scores):
    order = np.argsort(-scores, kind="stable")

    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)

    return ranks


def evaluate_rank(ranks, relevant_indices):
    best = int(np.min(ranks[relevant_indices]))

    return {
        "first_relevant_rank": best,
        "hit_at_1": int(best <= 1),
        "hit_at_5": int(best <= 5),
        "hit_at_10": int(best <= 10),
        "reciprocal_rank": 1.0 / best,
    }


def summarize(rows):
    frame = pd.DataFrame(rows)

    return {
        "Queries": len(frame),
        "Hit@1": float(frame["hit_at_1"].mean()),
        "Hit@5": float(frame["hit_at_5"].mean()),
        "Hit@10": float(frame["hit_at_10"].mean()),
        "MRR": float(frame["reciprocal_rank"].mean()),
    }


def evaluate_all(issues, eligible, relevant, matrix, embeddings):
    issue_ids = issues["issue_id"].to_numpy()

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    id_to_idx = {
        issue_id: i
        for i, issue_id in enumerate(issue_ids)
    }

    configurations = {
        f"{'full' if n == 0 else f'top{n}'}_w{int(w * 100)}": {
            "top_n": n,
            "semantic_weight": w,
        }
        for n in TOP_NS
        for w in WEIGHTS
    }

    records = {
        "tfidf": [],
        "minilm": [],
        **{name: [] for name in configurations},
    }

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc="Evaluating fusion",
    ):
        query_idx = id_to_idx[query.issue_id]

        cutoff = np.searchsorted(
            timestamps,
            np.datetime64(query.created_at.to_datetime64()),
            side="left",
        )

        if cutoff == 0:
            raise ValueError("No earlier candidates.")

        lexical_scores = (
            matrix[:cutoff] @ matrix[query_idx].T
        ).toarray().ravel()

        semantic_scores = np.asarray(
            embeddings[:cutoff] @ embeddings[query_idx]
        )

        lexical_ranks = get_ranks(lexical_scores)
        semantic_ranks = get_ranks(semantic_scores)

        relevant_indices = np.array([
            id_to_idx[rid]
            for rid in relevant[query.issue_id]
            if rid in id_to_idx and id_to_idx[rid] < cutoff
        ], dtype=int)

        if not len(relevant_indices):
            raise ValueError(
                f"No earlier duplicate for {query.issue_id}"
            )

        for name, ranks in (
            ("tfidf", lexical_ranks),
            ("minilm", semantic_ranks),
        ):
            result = evaluate_rank(
                ranks, relevant_indices
            )

            records[name].append({
                "query_id": query.issue_id,
                **result,
            })

        for name, config in configurations.items():
            n = config["top_n"]
            weight = config["semantic_weight"]

            lexical_contribution = (
                (1.0 - weight)
                / (RRF_K + lexical_ranks)
            )

            semantic_contribution = (
                weight
                / (RRF_K + semantic_ranks)
            )

            if n:
                lexical_contribution = np.where(
                    lexical_ranks <= n,
                    lexical_contribution,
                    0.0,
                )

                semantic_contribution = np.where(
                    semantic_ranks <= n,
                    semantic_contribution,
                    0.0,
                )

            fused_scores = (
                lexical_contribution
                + semantic_contribution
            )

            fused_ranks = get_ranks(fused_scores)

            if n and not np.any(
                fused_scores[relevant_indices] > 0
            ):
                result = {
                    "first_relevant_rank": cutoff + 1,
                    "hit_at_1": 0,
                    "hit_at_5": 0,
                    "hit_at_10": 0,
                    "reciprocal_rank": 0.0,
                }
            else:
                result = evaluate_rank(
                    fused_ranks, relevant_indices
                )

            records[name].append({
                "query_id": query.issue_id,
                **result,
            })

    summary = []

    for name, rows in records.items():
        summary.append({
            "configuration": name,
            **summarize(rows),
        })

    return (
        pd.DataFrame(summary)
        .sort_values("MRR", ascending=False)
        .reset_index(drop=True),
        records,
    )


def main(args):
    if args.split != "validation":
        raise ValueError(
            "Fusion development is validation-only."
        )

    issues, queries, eligible, relevant = prepare_benchmark(
        args.dataset, args.split
    )

    print("Dataset:", args.dataset)
    print("Model:", args.model)
    print("Eligible queries:", len(eligible))

    cutoff = queries["created_at"].min()

    fitting = issues[
        issues["created_at"] < cutoff
    ]

    print("TF-IDF fitting reports:", len(fitting))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )

    vectorizer.fit(fitting["text"].tolist())

    matrix = vectorizer.transform(
        issues["text"].tolist()
    )

    embeddings = load_or_create_embeddings(
        issues,
        args.model,
        args.dataset,
        args.split,
        args.batch_size,
    )

    summary, records = evaluate_all(
        issues, eligible, relevant, matrix, embeddings
    )

    print("\nFusion comparison")
    print(summary.to_string(index=False, float_format="%.4f"))

    RESULTS.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.dataset}_{args.model}_fusion_validation"

    summary.to_csv(
        RESULTS / f"{prefix}_summary.csv",
        index=False,
    )

    for name, rows in records.items():
        pd.DataFrame(rows).to_csv(
            RESULTS / f"{prefix}_{name}_queries.csv",
            index=False,
        )

    pd.DataFrame(records["minilm"]).to_csv(
    RESULTS / f"{prefix}_minilm_queries.csv",
    index=False,
)

    best = summary.iloc[0]["configuration"]

    pd.DataFrame(records[best]).to_csv(
        RESULTS / f"{prefix}_best_queries.csv",
        index=False,
    )

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "rrf_k": RRF_K,
        "weights": WEIGHTS,
        "top_ns": TOP_NS,
        "best_by_validation_mrr": best,
    }

    with open(
        RESULTS / f"{prefix}_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\nBest by validation MRR:", best)
    print("Results saved:", prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_PATHS.keys()),
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["validation"],
        default="validation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    main(parser.parse_args())

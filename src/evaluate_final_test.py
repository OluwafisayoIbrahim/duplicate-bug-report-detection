
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from src.evaluate import prepare_benchmark
from src.evaluate_hybrid import (
    MODEL_PATHS,
    get_ranks,
    load_model,
)
from src.evaluate_entity_retrieval import (
    load_entities,
    jaccard,
)
from src.evaluate_entity_idf import (
    build_idf_weights,
    weighted_jaccard,
    rerank_with_scores,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/final_test.json"
OUTPUT = ROOT / "results/final_test"
EMBEDDING_CACHE = ROOT / "models/final_test_cache"

EXPECTED_CONFIG_SHA = (
    "ef05f3bed51bd4124460b838498fca8665599e361"
    "e4ce8a7c6215558b588d978"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def hash_directory(path):
    digest = hashlib.sha256()

    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue

        relative = file.relative_to(path)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file).encode("ascii"))

    return digest.hexdigest()


def load_config():
    raw = CONFIG_PATH.read_bytes()
    signature = hashlib.sha256(raw).hexdigest()

    if signature != EXPECTED_CONFIG_SHA:
        raise RuntimeError(
            "Frozen configuration has changed. "
            "Stop before evaluating the test set."
        )

    config = json.loads(raw)

    if config["split"] != "test":
        raise ValueError("Expected test split.")

    if config["allow_test_parameter_tuning"]:
        raise ValueError("Test tuning is prohibited.")

    if config["fusion"]["semantic_weight"] != 0.95:
        raise ValueError("Unexpected fusion weight.")

    if config["fusion"]["rrf_k"] != 60:
        raise ValueError("Unexpected RRF constant.")

    if config["entity_reranking"]["weight"] != 0.30:
        raise ValueError("Unexpected entity weight.")

    if config["entity_reranking"]["top_k"] != 50:
        raise ValueError("Unexpected reranking window.")

    return config, signature


def repository(issue_id):
    value = str(issue_id)

    if "#" not in value:
        raise ValueError(f"Invalid repository issue ID: {value}")

    return value.rsplit("#", 1)[0].lower()


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def get_embedding_cache(
    dataset,
    model_name,
    issues,
    model_hash,
    config_hash,
):
    EMBEDDING_CACHE.mkdir(parents=True, exist_ok=True)

    path = EMBEDDING_CACHE / f"{dataset}_test_embeddings.npy"
    metadata_path = EMBEDDING_CACHE / f"{dataset}_test_metadata.json"

    digest = hashlib.sha256()
    digest.update(model_hash.encode("ascii"))
    digest.update(config_hash.encode("ascii"))

    for row in issues[
        ["issue_id", "created_at", "text"]
    ].itertuples(index=False, name=None):
        for value in row:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")

    signature = digest.hexdigest()

    if path.exists() and metadata_path.exists():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if metadata.get("signature") == signature:
            embeddings = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

            expected_shape = (
                len(issues),
                384,
            )

            if (
                embeddings.shape == expected_shape
                and sha256_file(path)
                == metadata.get("file_sha256")
            ):
                print("Loaded verified test embedding cache.")
                return embeddings, path

    print("Encoding test benchmark issues...")

    model = load_model(model_name)
    model.max_seq_length = 256

    vectors = model.encode(
        issues["text"].fillna("").tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.shape != (len(issues), 384):
        raise ValueError(
            f"Unexpected embedding shape: {vectors.shape}"
        )

    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings are not normalized.")

    np.save(path, vectors)

    metadata = {
        "dataset": dataset,
        "model": model_name,
        "signature": signature,
        "shape": list(vectors.shape),
        "file_sha256": sha256_file(path),
        "normalized": True,
        "max_seq_length": 256,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return np.load(
        path, mmap_mode="r", allow_pickle=False
    ), path


def prepare_data(dataset):
    issues, queries, eligible, relevant = (
        prepare_benchmark(dataset, split="test")
    )

    issues = issues.copy()

    ids = issues["issue_id"].astype(str).to_numpy()

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    # The benchmark's actual outer test boundary.
    cutoff_time = queries["created_at"].min()

    fitting_mask = (
        issues["created_at"] < cutoff_time
    ).to_numpy()

    fitting_indices = np.flatnonzero(fitting_mask)

    if len(fitting_indices) == 0:
        raise ValueError("Empty fitting period.")

    if not np.all(fitting_mask[:len(fitting_indices)]):
        raise ValueError("Fitting period is not contiguous.")

    if fitting_mask[len(fitting_indices):].any():
        raise ValueError("Fitting boundary is inconsistent.")

    if not (
        eligible["created_at"] >= cutoff_time
    ).all():
        raise ValueError("Eligible query before test cutoff.")

    return (
        issues,
        queries,
        eligible,
        relevant,
        ids,
        timestamps,
        cutoff_time,
        fitting_indices,
    )


def audit_dataset(dataset, prepared):
    (
        issues,
        queries,
        eligible,
        relevant,
        ids,
        timestamps,
        cutoff_time,
        fitting_indices,
    ) = prepared

    id_to_idx = {
        issue_id: i for i, issue_id in enumerate(ids)
    }

    covered = 0
    unavailable = []
    candidate_counts = []

    for query in eligible.itertuples(index=False):
        query_id = str(query.issue_id)
        query_idx = id_to_idx[query_id]

        query_timestamp = timestamps[query_idx]

        cutoff = np.searchsorted(
            timestamps,
            query_timestamp,
            side="left",
        )

        if cutoff <= 0:
            raise ValueError(f"No candidates: {query_id}")

        if not np.all(
            timestamps[:cutoff] < query_timestamp
        ):
            raise ValueError("Chronological candidate violation.")

        relevant_ids = relevant[query_id]

        for duplicate_id in relevant_ids:
            idx = id_to_idx[str(duplicate_id)]

            if idx >= cutoff:
                raise ValueError(
                    f"Future duplicate detected: {query_id}"
                )

        if dataset == "issuedupes":
            query_repo = repository(query_id)

            repos = np.array(
                [repository(ids[i]) for i in range(cutoff)]
            )

            local_indices = np.flatnonzero(
                repos == query_repo
            )

            candidate_counts.append(len(local_indices))

            same_repo_relevant = [
                rid for rid in relevant_ids
                if repository(rid) == query_repo
            ]

            if same_repo_relevant:
                covered += 1
            else:
                unavailable.append(query_id)

    report = {
        "dataset": dataset,
        "total_reports": len(issues),
        "development_reports": len(fitting_indices),
        "test_period_reports": len(queries),
        "eligible_test_queries": len(eligible),
        "test_cutoff": str(cutoff_time),
    }

    if dataset == "issuedupes":
        report.update({
            "same_repo_covered": covered,
            "same_repo_uncovered": len(unavailable),
            "same_repo_coverage_rate": (
                covered / len(eligible)
                if len(eligible) else 0.0
            ),
            "mean_same_repo_candidates": (
                float(np.mean(candidate_counts))
                if candidate_counts else 0.0
            ),
            "uncovered_query_ids": unavailable,
        })

    print("\n" + "=" * 65)
    print("TEST PROTOCOL AUDIT:", dataset)
    print("=" * 65)

    for key, value in report.items():
        if key != "uncovered_query_ids":
            print(f"{key}: {value}")

    return report


def calculate_metrics(ranks, relevant_local):
    best = int(np.min(ranks[relevant_local]))

    return {
        "first_relevant_rank": best,
        "hit_at_1": int(best <= 1),
        "hit_at_5": int(best <= 5),
        "hit_at_10": int(best <= 10),
        "reciprocal_rank": 1.0 / best,
    }


def evaluate_dataset(
    dataset,
    settings,
    config,
    config_hash,
    prepared,
):
    (
        issues,
        queries,
        eligible,
        relevant,
        ids,
        timestamps,
        cutoff_time,
        fitting_indices,
    ) = prepared

    model_name = settings["model_name"]

    model_path = ROOT / settings["model_path"]

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    configured_model = MODEL_PATHS[model_name]

    if Path(configured_model).resolve() != model_path.resolve():
        raise ValueError(
            "Configured checkpoint does not match MODEL_PATHS."
        )

    model_hash = hash_directory(model_path)

    embeddings, embedding_path = get_embedding_cache(
        dataset,
        model_name,
        issues,
        model_hash,
        config_hash,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )

    vectorizer.fit(
        issues.iloc[fitting_indices]["text"].tolist()
    )

    tfidf = vectorizer.transform(
        issues["text"].tolist()
    )

    entities = load_entities(
        issues, dataset=dataset
    )

    idf_weights, unseen_weight = build_idf_weights(
        entities,
        fitting_indices,
    )

    id_to_idx = {
        issue_id: i for i, issue_id in enumerate(ids)
    }

    repositories = (
        np.array([repository(x) for x in ids])
        if dataset == "issuedupes"
        else None
    )

    records = []
    methods = settings["methods"]

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc=f"Final test: {dataset}",
    ):
        query_id = str(query.issue_id)
        query_idx = id_to_idx[query_id]

        cutoff = np.searchsorted(
            timestamps,
            timestamps[query_idx],
            side="left",
        )

        relevant_ids = set(
            str(x) for x in relevant[query_id]
        )

        modes = settings["candidate_modes"]

        for mode in modes:
            if mode == "unrestricted":
                candidate_indices = np.arange(cutoff)

            elif mode == "same_repository":
                query_repo = repositories[query_idx]

                candidate_indices = np.flatnonzero(
                    repositories[:cutoff] == query_repo
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            if len(candidate_indices) == 0:
                continue

            relevant_local = np.array(
                [
                    i for i, global_idx
                    in enumerate(candidate_indices)
                    if ids[global_idx] in relevant_ids
                ],
                dtype=np.int64,
            )

            # A restricted query without an earlier
            # same-repo duplicate is unreachable.
            # Coverage is reported separately.
            if len(relevant_local) == 0:
                if mode == "unrestricted":
                    raise ValueError(
                        f"Missing unrestricted label: {query_id}"
                    )
                continue

            lexical_scores = (
                tfidf[candidate_indices]
                @ tfidf[query_idx].T
            ).toarray().ravel()

            semantic_scores = np.asarray(
                embeddings[candidate_indices]
                @ embeddings[query_idx]
            ).ravel()

            lexical_ranks = get_ranks(lexical_scores)
            semantic_ranks = get_ranks(semantic_scores)

            fusion = config["fusion"]

            fused_scores = (
                fusion["lexical_weight"]
                / (fusion["rrf_k"] + lexical_ranks)
                + fusion["semantic_weight"]
                / (fusion["rrf_k"] + semantic_ranks)
            )

            hybrid_ranks = get_ranks(fused_scores)

            rankings = {
                "minilm": semantic_ranks,
                "hybrid": hybrid_ranks,
            }

            if any("hybrid_w0.3" in m for m in methods):
                n = min(
                    config["entity_reranking"]["top_k"],
                    len(candidate_indices),
                )

                top_local = np.argsort(
                    hybrid_ranks,
                    kind="stable",
                )[:n]

                query_entities = entities[query_idx]

                ordinary = np.array([
                    jaccard(
                        query_entities,
                        entities[int(candidate_indices[i])],
                    )
                    for i in top_local
                ])

                weighted = np.array([
                    weighted_jaccard(
                        query_entities,
                        entities[int(candidate_indices[i])],
                        idf_weights,
                        unseen_weight,
                    )
                    for i in top_local
                ])

                weight = config[
                    "entity_reranking"
                ]["weight"]

                rankings["jaccard_hybrid_w0.3"] = (
                    rerank_with_scores(
                        hybrid_ranks,
                        ordinary,
                        weight,
                    )
                )

                rankings["idf_hybrid_w0.3"] = (
                    rerank_with_scores(
                        hybrid_ranks,
                        weighted,
                        weight,
                    )
                )

            for method in methods:
                result = calculate_metrics(
                    rankings[method],
                    relevant_local,
                )

                records.append({
                    "dataset": dataset,
                    "query_id": query_id,
                    "group_id": str(
                        issues.iloc[query_idx]["group_id"]
                    ),
                    "mode": mode,
                    "method": method,
                    "candidate_count": len(candidate_indices),
                    **result,
                })

    frame = pd.DataFrame(records)

    if frame.duplicated(
        ["dataset", "query_id", "mode", "method"]
    ).any():
        raise ValueError("Duplicate evaluation records.")

    manifest = {
        "dataset": dataset,
        "model_path": str(settings["model_path"]),
        "model_sha256": model_hash,
        "embedding_cache": str(
            embedding_path.relative_to(ROOT)
        ),
        "embedding_sha256": sha256_file(embedding_path),
        "test_cutoff": str(cutoff_time),
        "fitting_reports": len(fitting_indices),
        "issues_sha256": sha256_file(
            ROOT / "data/processed"
            / f"{dataset}_issues_grouped.csv"
        ),
        "links_sha256": sha256_file(
            ROOT / "data/processed"
            / f"{dataset}_direct_links.csv"
        ),
    }

    return frame, manifest


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--audit-only",
        action="store_true",
    )

    mode.add_argument(
        "--run-final",
        action="store_true",
    )

    args = parser.parse_args()

    config, config_hash = load_config()

    if args.run_final:
        for dataset in config["datasets"]:
            output_file = (
                OUTPUT / f"{dataset}_final_queries.csv"
            )

            if output_file.exists():
                raise RuntimeError(
                    f"Final results already exist: {output_file}"
                )

    prepared_data = {}
    audit_reports = {}

    for dataset in config["datasets"]:
        prepared = prepare_data(dataset)

        prepared_data[dataset] = prepared

        audit_reports[dataset] = audit_dataset(
            dataset, prepared
        )

    if args.audit_only:
        print("\nAudit complete. No model scoring executed.")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)

    manifests = {}

    for dataset, settings in config["datasets"].items():
        frame, manifest = evaluate_dataset(
            dataset,
            settings,
            config,
            config_hash,
            prepared_data[dataset],
        )

        output_file = (
            OUTPUT / f"{dataset}_final_queries.csv"
        )

        frame.to_csv(output_file, index=False)

        summary = (
            frame.groupby(
                ["dataset", "mode", "method"],
                sort=False,
            )
            .agg(
                queries=("query_id", "size"),
                Hit1=("hit_at_1", "mean"),
                Hit5=("hit_at_5", "mean"),
                Hit10=("hit_at_10", "mean"),
                MRR=("reciprocal_rank", "mean"),
                mean_candidates=(
                    "candidate_count", "mean"
                ),
            )
            .reset_index()
        )

        summary.to_csv(
            OUTPUT / f"{dataset}_final_summary.csv",
            index=False,
        )

        print("\n", dataset, "final test summary")
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        manifests[dataset] = manifest

    final_manifest = {
        "experiment": config["experiment"],
        "config_sha256": config_hash,
        "git_revision": git_revision(),
        "status": "test_evaluation_completed",
        "audits": audit_reports,
        "datasets": manifests,
    }

    (OUTPUT / "final_test_manifest.json").write_text(
        json.dumps(final_manifest, indent=2),
        encoding="utf-8",
    )

    print("\nFinal test evaluation completed.")
    print("Results directory:", OUTPUT)


if __name__ == "__main__":
    main()

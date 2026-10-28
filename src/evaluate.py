
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_PATHS = {
    "issuedupes": {
        "pretrained": BASE_MODEL,
        "pilot": ROOT / "models" / "issuedupes_minilm_pilot" / "final",
        "1epoch": ROOT / "models" / "issuedupes_minilm_1epoch" / "final",
    },
    "eclipse": {
    "pretrained": BASE_MODEL,
    "pilot": ROOT / "models" / "eclipse_minilm_pilot" / "final",
    "1epoch": ROOT / "models" / "eclipse_minilm_1epoch" / "final",
},
}

DATASET_FILES = {
    "issuedupes": {
        "issues": "issuedupes_issues_grouped.csv",
        "links": "issuedupes_direct_links.csv",
    },
    "eclipse": {
        "issues": "eclipse_issues_grouped.csv",
        "links": "eclipse_direct_links.csv",
    },
}


def prepare_validation(dataset):
    files = DATASET_FILES[dataset]

    issues = pd.read_csv(
        PROCESSED / files["issues"],
        dtype={"issue_id": str},
        low_memory=False,
    )

    links = pd.read_csv(
        PROCESSED / files["links"],
        dtype={"source_id": str, "target_id": str},
    )

    issues["issue_id"] = issues["issue_id"].astype(str)

    issues["created_at"] = pd.to_datetime(
        issues["created_at"],
        utc=True,
        errors="coerce",
    )

    issues["text"] = issues["text"].fillna("").astype(str)

    issues = issues[
        issues["text"].str.strip().ne("")
        & issues["created_at"].notna()
    ].copy()

    issues = issues.sort_values(
        ["created_at", "issue_id"]
    ).reset_index(drop=True)

    outer_idx = int(len(issues) * 0.8)
    outer_cutoff = issues.iloc[outer_idx]["created_at"]

    training_period = issues[
        issues["created_at"] < outer_cutoff
    ].copy().reset_index(drop=True)

    inner_idx = int(len(training_period) * 0.85)
    inner_cutoff = training_period.iloc[inner_idx]["created_at"]

    validation = training_period[
        training_period["created_at"] >= inner_cutoff
    ].copy()

    valid_ids = set(training_period["issue_id"])
    adjacency = {}

    for source, target in links[
        ["source_id", "target_id"]
    ].itertuples(index=False, name=None):

        if (
            source == target
            or source not in valid_ids
            or target not in valid_ids
        ):
            continue

        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    created_lookup = training_period.set_index(
        "issue_id"
    )["created_at"].to_dict()

    relevant = {}

    for row in validation.itertuples(index=False):
        earlier = {
            other
            for other in adjacency.get(row.issue_id, set())
            if created_lookup[other] < row.created_at
        }

        if earlier:
            relevant[row.issue_id] = earlier

    eligible = validation[
        validation["issue_id"].isin(relevant)
    ].copy()

    return training_period, validation, eligible, relevant


def evaluate_retrieval(issues, eligible, relevant, embeddings):
    issue_ids = issues["issue_id"].to_numpy()

    timestamps = issues["created_at"].to_numpy(
        dtype="datetime64[ns]"
    )

    id_to_idx = {
        issue_id: idx
        for idx, issue_id in enumerate(issue_ids)
    }

    ranks = []

    for query in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc="Evaluating retrieval",
    ):
        query_idx = id_to_idx[query.issue_id]

        cutoff = np.searchsorted(
            timestamps,
            np.datetime64(query.created_at.to_datetime64()),
            side="left",
        )

        if cutoff == 0:
            continue

        scores = np.asarray(
            embeddings[:cutoff] @ embeddings[query_idx]
        )

        relevant_indices = [
            id_to_idx[rid]
            for rid in relevant[query.issue_id]
            if rid in id_to_idx and id_to_idx[rid] < cutoff
        ]

        if not relevant_indices:
            raise ValueError(
                f"No eligible relevant report for {query.issue_id}"
            )

        best_score = max(
            scores[idx] for idx in relevant_indices
        )

        best_idx = min(
            idx
            for idx in relevant_indices
            if scores[idx] == best_score
        )

        rank = (
            np.count_nonzero(scores > best_score)
            + np.count_nonzero(scores[:best_idx] == best_score)
            + 1
        )

        ranks.append(rank)

    ranks = np.asarray(ranks)

    if len(ranks) == 0:
        raise ValueError("No eligible queries were evaluated.")

    return {
        "Queries": len(ranks),
        "Hit@1": float(np.mean(ranks <= 1)),
        "Hit@5": float(np.mean(ranks <= 5)),
        "Hit@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
    }

def load_model(dataset, model_name):
    model_path = MODEL_PATHS[dataset].get(model_name)

    if model_path is None:
        raise ValueError(
            f"Model '{model_name}' is not configured "
            f"for dataset '{dataset}'."
        )

    if isinstance(model_path, Path) and not model_path.is_dir():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    print(f"\nLoading model: {model_name}")
    print(f"Model source: {model_path}")

    model = SentenceTransformer(
        str(model_path),
        device="cpu",
    )

    model.max_seq_length = 256

    return model



def main(args):
    print("=" * 55)
    print("Duplicate Bug Report Retrieval Evaluation")
    print("=" * 55)

    print("Dataset:", args.dataset)
    print("Model:", args.model)
    print("Evaluation split: validation")

    issues, validation, eligible, relevant = prepare_validation(
        args.dataset
    )

    print("\nBenchmark statistics")
    print("Training-period reports:", len(issues))
    print("Validation reports:", len(validation))
    print("Eligible validation queries:", len(eligible))

    if not len(eligible):
        raise ValueError(
            "No eligible validation queries found."
        )

    model = load_model(args.dataset, args.model)

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        issues["text"].tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("Embedding shape:", embeddings.shape)

    if embeddings.shape[0] != len(issues):
        raise ValueError("Embedding count does not match issue count.")

    print("\nEvaluating retrieval...")

    results = evaluate_retrieval(
        issues=issues,
        eligible=eligible,
        relevant=relevant,
        embeddings=embeddings,
    )

    print("\nResults")
    print("-" * 35)

    for metric, value in results.items():
        if metric == "Queries":
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    output_path = (
        RESULTS
        / f"{args.dataset}_{args.model}_validation.csv"
    )

    results_row = {
        "Dataset": args.dataset,
        "Model": args.model,
        "Split": "validation",
        **results,
    }

    pd.DataFrame([results_row]).to_csv(
        output_path,
        index=False,
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate duplicate bug report retrieval."
    )

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
        help="Dataset to evaluate.",
    )

    parser.add_argument(
        "--model",
        choices=["pretrained", "pilot", "1epoch"],
        required=True,
        help="Model checkpoint to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Sentence-BERT inference batch size.",
    )

    main(parser.parse_args())

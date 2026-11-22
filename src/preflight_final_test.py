
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "final_test.json"

REQUIRED_SCRIPTS = [
    "src/evaluate.py",
    "src/evaluate_hybrid.py",
    "src/evaluate_entity_retrieval.py",
    "src/evaluate_entity_idf.py",
    "src/evaluate_repository_entity.py",
    "src/extract_technical_entities.py",
]

REQUIRED_VALIDATION_RESULTS = [
    "results/eclipse_entity_idf_queries.csv",
    "results/eclipse_entity_idf_summary.csv",
    "results/issuedupes_entity_idf_queries.csv",
    "results/issuedupes_entity_idf_summary.csv",
    "results/issuedupes_repository_entity_queries.csv",
]

REQUIRED_DATA_FILES = [
    "data/processed/eclipse_issues_grouped.csv",
    "data/processed/eclipse_direct_links.csv",
    "data/processed/issuedupes_issues_grouped.csv",
    "data/processed/issuedupes_direct_links.csv",
]


def check_path(relative_path, directory=False):
    path = ROOT / relative_path

    exists = (
        path.is_dir()
        if directory
        else path.is_file()
    )

    status = "OK" if exists else "MISSING"

    print(f"[{status}] {relative_path}")

    return exists


def main():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration: {CONFIG_PATH}"
        )

    raw_bytes = CONFIG_PATH.read_bytes()

    config = json.loads(
        raw_bytes.decode("utf-8")
    )

    signature = hashlib.sha256(
        raw_bytes
    ).hexdigest()

    print("\nFINAL TEST PREFLIGHT")
    print("=" * 65)

    print("Configuration:", CONFIG_PATH)
    print("SHA-256:", signature)

    if config["split"] != "test":
        raise ValueError("Final split must be test.")

    if config["allow_test_parameter_tuning"]:
        raise ValueError(
            "Test parameter tuning must be disabled."
        )

    if config["candidate_rule"] != (
        "strictly_earlier_than_query"
    ):
        raise ValueError(
            "Unexpected candidate eligibility rule."
        )

    if config["entity_reranking"]["idf_fit_scope"] != (
        "outer_development_only"
    ):
        raise ValueError(
            "Entity IDF must use development data."
        )

    if config["fusion"]["semantic_weight"] != 0.95:
        raise ValueError(
            "Unexpected semantic fusion weight."
        )

    if config["entity_reranking"]["weight"] != 0.30:
        raise ValueError(
            "Unexpected entity reranking weight."
        )

    if config["entity_reranking"]["top_k"] != 50:
        raise ValueError(
            "Unexpected entity reranking window."
        )

    missing = []

    print("\nChecking scripts")
    print("-" * 65)

    for path in REQUIRED_SCRIPTS:
        if not check_path(path):
            missing.append(path)

    print("\nChecking datasets")
    print("-" * 65)

    for path in REQUIRED_DATA_FILES:
        if not check_path(path):
            missing.append(path)

    print("\nChecking validation artifacts")
    print("-" * 65)

    for path in REQUIRED_VALIDATION_RESULTS:
        if not check_path(path):
            missing.append(path)

    print("\nChecking frozen models")
    print("-" * 65)

    for dataset, settings in config["datasets"].items():
        model_path = settings["model_path"]

        if not check_path(model_path, directory=True):
            missing.append(model_path)

    print("\nChecking Python dependencies")
    print("-" * 65)

    import numpy
    import pandas
    import sklearn
    import sentence_transformers

    print("NumPy:", numpy.__version__)
    print("Pandas:", pandas.__version__)
    print("scikit-learn:", sklearn.__version__)
    print(
        "Sentence Transformers:",
        sentence_transformers.__version__,
    )

    if missing:
        print("\nPreflight FAILED.")
        print("Missing items:", len(missing))

        raise SystemExit(1)

    manifest = {
        "experiment": config["experiment"],
        "configuration": str(
            CONFIG_PATH.relative_to(ROOT)
        ),
        "configuration_sha256": signature,
        "status": "preflight_passed",
        "test_evaluation_executed": False,
    }

    output_dir = (
        ROOT / config["output_directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir / "experiment_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("PREFLIGHT PASSED")
    print("=" * 65)

    print("Configuration SHA-256:", signature)
    print("Manifest:", manifest_path)

    print(
        "\nNo held-out test evaluation "
        "has been executed."
    )


if __name__ == "__main__":
    main()

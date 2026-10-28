import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.sentence_transformer.training_args import BatchSamplers


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATASETS = {
    "issuedupes": "issuedupes_fit_pairs.csv",
    "eclipse": "eclipse_fit_pairs.csv",
}


def select_group_safe_pairs(pairs):
    """Select one positive pair per training-only duplicate group."""

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for source, target in pairs[
        ["source_id", "target_id"]
    ].itertuples(index=False, name=None):
        union(str(source), str(target))

    pairs = pairs.copy()

    pairs["train_group"] = pairs["source_id"].map(find)

    return (
        pairs.sort_values(["source_id", "target_id"])
        .drop_duplicates(subset=["train_group"])
        .reset_index(drop=True)
    )


def main(args):
    pairs = pd.read_csv(
        PROCESSED / DATASETS[args.dataset],
        dtype={"source_id": str, "target_id": str},
    )

    pairs = select_group_safe_pairs(pairs)

    train_dataset = Dataset.from_dict({
        "anchor": pairs["text_a"].astype(str).tolist(),
        "positive": pairs["text_b"].astype(str).tolist(),
    })

    print("Dataset:", args.dataset)
    print("Training examples:", len(train_dataset))

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    model.max_seq_length = 256

    loss = losses.MultipleNegativesRankingLoss(model)

    output_dir = (
        ROOT
        / "models"
        / f"{args.dataset}_minilm_{args.run_name}"
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        use_cpu=True,
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()

    final_path = output_dir / "final"
    model.save(str(final_path))

    print("Model saved to:", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="1epoch",
    )

    main(parser.parse_args())
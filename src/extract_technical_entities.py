
import argparse
import json
import re
from collections import Counter

import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


PATTERNS = {
    "exceptions": re.compile(
        r"\b[A-Za-z_$][\w$]*"
        r"(?:Exception|Error)\b"
    ),

    "qualified_identifiers": re.compile(
        r"\b(?:[A-Za-z_$][\w$]*\.){2,}"
        r"[A-Za-z_$][\w$]*\b"
    ),

    "file_names": re.compile(
        r"\b[\w.-]+\."
        r"(?:xml|json|yaml|yml|toml|properties|"
        r"java|py|js|jsx|ts|tsx|cs|cpp|html|css|md)\b",
        re.IGNORECASE,
    ),

    "method_calls": re.compile(
        r"\b[A-Za-z_$][\w$]*\s*\(\s*\)"
    ),

    "config_keys": re.compile(
        r"\b(?:[a-z]+(?:_[a-z0-9]+)+|"
        r"[a-z]+(?:[A-Z][a-zA-Z0-9]*)+)\b"
    ),

    "http_codes": re.compile(
        r"\bHTTP\s+[45]\d{2}\b",
        re.IGNORECASE,
    ),
}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def extract_entities(text):
    text = clean_text(text)

    entities = {}

    for category, pattern in PATTERNS.items():
        matches = pattern.findall(text)

        normalized = {
            re.sub(r"\s+", "", match).lower()
            for match in matches
        }

        entities[category] = sorted(normalized)

    return entities


def flatten_entities(entities):
    return {
        item
        for values in entities.values()
        for item in values
    }


def analyze_dataset(dataset, sample_size=1000):
    path = (
        PROCESSED
        / f"{dataset}_issues_grouped.csv"
    )

    issues = pd.read_csv(
        path,
        dtype={"issue_id": str},
        usecols=["issue_id", "text"],
        low_memory=False,
    )

    sample = issues.sample(
        n=min(sample_size, len(issues)),
        random_state=42,
    )

    counts = Counter()
    examples = {}

    for row in sample.itertuples(index=False):
        entities = extract_entities(row.text)

        for category, values in entities.items():
            counts[category] += len(values)

            if values and category not in examples:
                examples[category] = {
                    "issue_id": row.issue_id,
                    "entities": values[:10],
                }

    print("\nDataset:", dataset)
    print("Sampled reports:", len(sample))

    print("\nEntity counts")
    print("-" * 45)

    for category in PATTERNS:
        print(
            f"{category}: {counts[category]}"
        )

    print("\nExamples")
    print("-" * 45)

    for category, example in examples.items():
        print(f"\n{category}")
        print("Issue:", example["issue_id"])
        print("Entities:", example["entities"])


def main(args):
    if args.text is not None:
        entities = extract_entities(args.text)

        print(
            json.dumps(
                entities,
                indent=2,
                ensure_ascii=False,
            )
        )

    elif args.issue_id:
        issues = pd.read_csv(
            PROCESSED / f"{args.dataset}_issues_grouped.csv",
            dtype={"issue_id": str},
            low_memory=False,
        )

        match = issues[
            issues["issue_id"] == args.issue_id
        ]

        if match.empty:
            raise ValueError(
                f"Issue not found: {args.issue_id}"
            )

        entities = extract_entities(
            match.iloc[0]["text"]
        )

        print("Issue:", args.issue_id)
        print(
            json.dumps(
                entities,
                indent=2,
                ensure_ascii=False,
            )
        )

    else:
        analyze_dataset(
            args.dataset,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["issuedupes", "eclipse"],
        default="eclipse",
    )

    parser.add_argument("--issue-id")

    parser.add_argument("--text")

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
    )

    main(parser.parse_args())

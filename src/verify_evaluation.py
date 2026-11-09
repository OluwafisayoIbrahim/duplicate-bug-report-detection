
from src.evaluate import prepare_benchmark


def verify(dataset, split):
    issues, queries, eligible, relevant = prepare_benchmark(
        dataset, split
    )

    timestamps = issues.set_index("issue_id")[
        "created_at"
    ].to_dict()

    query_ids = set(queries["issue_id"])

    valid = True

    for query_id, relevant_ids in relevant.items():
        if query_id not in query_ids:
            valid = False
            print("Unexpected query:", query_id)

        if not relevant_ids:
            valid = False
            print("No relevant issues:", query_id)

        for candidate_id in relevant_ids:
            if candidate_id not in timestamps:
                valid = False
                print("Unknown candidate:", candidate_id)

            elif timestamps[candidate_id] >= timestamps[query_id]:
                valid = False
                print(
                    "Chronological violation:",
                    query_id,
                    candidate_id,
                )

    assert len(eligible) == len(relevant)

    print(
        f"{dataset} / {split}:",
        "PASS" if valid else "FAIL",
    )

    print("Available reports:", len(issues))
    print("Query-period reports:", len(queries))
    print("Eligible queries:", len(eligible))


for dataset in ("issuedupes", "eclipse"):
    for split in ("validation", "test"):
        verify(dataset, split)

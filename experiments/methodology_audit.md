# Methodology and Data Leakage Audit

## 1. Purpose

This audit evaluates the methodological consistency of the duplicate bug report retrieval experiments conducted using IssueDupesLite and Eclipse Platform.

The review focuses on chronological data splitting, training-data isolation, positive-pair integrity, retrieval evaluation, and reproducibility.

## 2. Chronological Split Verification

Both datasets were divided chronologically using an outer 80/20 training-development/test split and an inner 85/15 fitting/validation split.

The positional splitting method used during dataset preparation was compared against the timestamp-based splitting method used during evaluation.

**Results:**

| Partition | IssueDupesLite | Eclipse Platform |
|---|---:|---:|
| Training-development | 15,275 | 68,124 |
| Model fitting | 12,983 | 57,905 |
| Validation | 2,292 | 10,219 |
| Test | 3,819 | 17,031 |

All partition memberships matched exactly, with zero differences.

## 3. Training-Data Isolation

The saved positive training pairs were checked against the fitting, validation, and test partitions.

| Check | IssueDupesLite | Eclipse Platform |
|---|---:|---:|
| Saved positive pairs | 6,439 | 9,477 |
| Pair endpoints outside fitting period | 0 | 0 |
| Pair endpoints in validation | 0 | 0 |
| Pair endpoints in test | 0 | 0 |
| Empty-text pairs | 0 | 0 |
| Self-referencing pairs | 0 | 0 |

No direct training-pair overlap with validation or test reports was detected.

## 4. Positive-Pair Integrity

All saved positive pairs corresponded to recorded direct duplicate relationships.

IssueDupesLite contained 14 repeated unordered positive pairs before training selection. Eclipse contained none.

The group-safe training-pair selection procedure retained:

| Check | IssueDupesLite | Eclipse Platform |
|---|---:|---:|
| Selected training examples | 5,936 | 5,298 |
| Repeated unordered pairs | 0 | 0 |
| Repeated training groups | 0 | 0 |

The repeated IssueDupesLite relationships were not retained as duplicate examples in the selected training set.

## 5. Retrieval Evaluation

The evaluation protocol restricts candidate reports to those created strictly before each query.

Relevant reports are identified using recorded direct duplicate relationships.

The current benchmark preparation and chronological relevance checks passed for all four evaluation configurations.

| Dataset | Validation eligible queries | Test eligible queries |
|---|---:|---:|
| IssueDupesLite | 1,236 | 2,258 |
| Eclipse Platform | 1,396 | 2,174 |

The evaluation measures retrieval effectiveness among queries with at least one known earlier direct duplicate. It does not measure end-to-end duplicate/nonduplicate classification.

## 6. Checkpoint Selection and Reproducibility

The available experiment history is consistent with model training and validation preceding held-out test evaluation.

The 50-step pilot checkpoints were initialized from pretrained MiniLM rather than continued from previously fine-tuned checkpoints.

The historical evaluation code in Git commit `4db4efa` contains chronological benchmark preparation, cross-dataset model configuration, and retrieval ranking.

However, Git history and filesystem timestamps alone do not conclusively establish that checkpoint selection was independent of all test feedback.

Exact reproduction of historical model metrics from a frozen code version remains a separate reproducibility check.

## 7. Remaining Methodological Limitations

- Duplicate annotations may be incomplete.
- Duplicate groups may span chronological partitions.
- Group identifiers are constructed retrospectively from recorded relationships.
- Evaluation includes only queries with known earlier direct duplicates.
- Some reports exceed the 256-token MiniLM sequence limit.
- Training used limited checkpoint configurations and random seeds.
- Available records do not conclusively establish the full checkpoint-selection chronology.

## 8. Conclusion

The audit found no direct training-pair overlap with validation or test reports and no chronological split-membership inconsistencies.

The saved positive pairs correspond to recorded duplicate links, and the selected training pairs contain no repeated unordered pairs or training groups.

These checks support the integrity of the audited data preparation and retrieval evaluation procedures. They do not establish that every possible form of data leakage has been eliminated.

The original experiments are retained as the baseline for subsequent hybrid retrieval research.
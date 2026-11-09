# Experimental Protocol — Original Baselines

## Research Task
Chronological retrieval of earlier, directly linked duplicate bug reports from software issue-tracking datasets.

## Datasets
- IssueDupesLite: 19,094 processed issue reports.
- Eclipse Platform: 85,155 processed issue reports.

## Data Splitting
- Outer split: earliest 80% for training/development, latest 20% for testing.
- Inner split: earliest 85% of the outer training period for model fitting, remaining 15% for validation.
- Chronological splits verified to have identical memberships across the data preparation and evaluation implementations.

## Retrieval Protocol
- Candidate reports must have been created strictly before the query.
- Relevant reports are identified using recorded direct duplicate relationships.
- Evaluation includes only queries with at least one known earlier duplicate.
- Metrics: Hit@1, Hit@5, Hit@10, and Mean Reciprocal Rank.

## Models
- TF-IDF lexical baseline.
- Pretrained `sentence-transformers/all-MiniLM-L6-v2`.
- Fine-tuned MiniLM using MultipleNegativesRankingLoss.
- Fine-tuning uses batch size 8, learning rate 2e-5, and a maximum sequence length of 256 tokens.
- Pilot checkpoints trained for 50 optimization steps from the pretrained model.

## Training Data Verification
- IssueDupesLite: 6,439 saved positive pairs, 5,936 selected training examples.
- Eclipse: 9,477 saved positive pairs, 5,298 selected training examples.
- No saved positive training-pair endpoints overlap with validation or test reports.
- Selected training examples have no repeated unordered pairs or training groups.

## Statistical Analysis
- Paired exact McNemar tests for Hit@K.
- Cluster-bootstrap confidence intervals for retrieval metric differences.
- Query-level ranking error analysis.
- Query-length and truncation analysis.

## Limitations
- Positive-query-only evaluation does not measure false-positive detection.
- Duplicate-link annotations may be incomplete.
- Connected duplicate groups can span chronological periods.
- Some reports exceed the 256-token model input limit.
- Reported experiments use limited checkpoint configurations and training seeds.

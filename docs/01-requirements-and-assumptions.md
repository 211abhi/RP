# Requirements and assumptions

## Confirmed requirements

- Reuse the architecture of the existing LLM-adapted time-series forecaster.
- Accept MIMC data as input.
- Tolerate missing columns.
- Complete documentation before implementation.
- Develop in a separate project folder.

## Working assumptions

These are placeholders, not facts about the dataset:

- Input is tabular, multivariate, and time-indexed.
- A file may contain one or many independent entities/series.
- Missing columns means a configured optional variable can be absent from an entire
  input, not merely contain null cells.
- The deployed model has a fixed canonical feature vocabulary. Incoming columns are
  mapped into that vocabulary; model dimensions never depend on a particular file.
- The original frozen-LLM, soft-prompt, contrastive-alignment, and prediction-head
  concepts remain. Dataset-specific feature engineering and semantic labels change.

## Open decisions

| ID | Decision | Why it blocks implementation |
|---|---|---|
| D1 | Exact meaning/version of MIMC | Determines tables, licensing, and schema |
| D2 | Entity ID and timestamp fields | Required for safe ordering and grouping |
| D3 | Target(s) and task type | Determines head, loss, metrics, and leakage rules |
| D4 | Sampling interval and horizon | Determines resampling, window, and output sizes |
| D5 | Canonical feature list | Determines adapter and tokenizer dimensions |
| D6 | Required vs optional columns | Determines reject/degrade behavior |
| D7 | Allowed imputation policy | Domain meaning and safety vary by variable |
| D8 | Split unit | Prevents entity leakage across splits |
| D9 | Semantic labels/context text | Replaces finance-specific regime descriptions |

## Safety and governance assumptions

If MIMC contains health or other sensitive records, raw identifiers must not appear in
logs, checkpoints, prompts, filenames, or model-generated text. Data access, retention,
de-identification, and downstream-use approval remain deployment responsibilities.
Any predictive output is research output until independently validated for its intended
use.

## Compatibility target

Components preserved conceptually:

- patch-based time-series tokenization;
- convolutional/Transformer time-series encoder;
- contrastive alignment to frozen LLM text embeddings;
- gradient-free soft-prompt optimization;
- frozen causal LLM;
- small trainable prediction head.

Components replaced or generalized:

- ETT downloader and synthetic OHLCV conversion;
- OHLCV-derived indicators;
- finance regime labeler and templates;
- fixed nine-feature assumptions;
- row-level chronological splitting when multiple entities exist.

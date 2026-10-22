# Implementation plan

Implementation starts only after the blocking decisions in the requirements document
are resolved.

## Phase 0: freeze the contract

- Obtain a de-identified schema/sample and data dictionary.
- Confirm task, target, horizon, entity/time fields, units, and feature availability.
- Complete the feature registry and leakage review.
- Define baseline metrics and deployment missingness scenarios.

Exit criterion: approved data contract and configuration with no `null` decision fields.

## Phase 1: data adapter and reports

- Implement CSV/Parquet loading, aliases, type/unit validation, entity grouping, duplicate
  handling, and canonical ordering.
- Implement entity-safe splits before fitting preprocessing state.
- Produce schema and quality reports.
- Add synthetic tests for absent columns, null cells, missing timestamps, duplicate times,
  bad types, and unknown aliases.

Exit criterion: deterministic tensors and leakage tests pass on a small fixture.

## Phase 2: mask-aware preprocessing/tokenizer

- Add causal per-feature imputation, masks, age features, and train-only scaling.
- Replace hardcoded feature counts and patch dimensions with registry-derived values.
- Add baseline concat fusion; add gated fusion only after baseline verification.
- Test that real zero, imputed zero, absent column, and padded step remain distinguishable.

Exit criterion: end-to-end window construction passes shape, causality, and serialization tests.

## Phase 3: semantic alignment adaptation

- Define and validate domain-relevant, pre-cutoff label generation.
- Replace finance templates and label counts.
- Train encoder alignment and compare with alignment-disabled ablation.

Exit criterion: alignment improves a predefined validation measure or is disabled.

## Phase 4: soft prompt and prediction head

- Reuse frozen LLM and CMA-ES design with checkpoint contracts.
- Implement the one selected task head and loss.
- Add deterministic resume and artifact metadata.

Exit criterion: overfit a tiny fixture, then train successfully on a bounded sample.

## Phase 5: evaluation

- Compare against naive, linear/tree, and non-LLM temporal baselines.
- Report task metrics overall and by missingness severity/column-availability pattern.
- Run feature-column dropout stress tests and temporal/entity leakage audits.
- Calibrate uncertainty or probabilities when relevant.

Exit criterion: acceptance criteria are met and limitations are documented.

## Phase 6: reproducible delivery

- Add a pipeline notebook/CLI, environment lock, artifact manifest, and model card.
- Document data preparation without bundling restricted data.
- Add CI for unit tests, config validation, and a tiny CPU smoke test.

## Migration map from the original repository

| Existing part | Action |
|---|---|
| `src/utils/data_loader.py` | Replace with registry-driven data package |
| patch tokenizer | Generalize for masks and computed dimensions |
| TS encoder | Reuse, adapting tokenizer input only |
| finance regime labeler/templates | Replace with approved domain label provider |
| frozen LLM wrapper | Reuse with contract validation |
| soft prompt/CMA-ES | Reuse with new data loader and artifact metadata |
| prediction head | Adapt to confirmed task/output shape |
| evaluation | Replace finance metrics/regime slices with task/missingness metrics |
| notebook | Rebuild after CLI pipeline is stable |

# Validation and acceptance

## Data tests

- Missing optional columns produce finite tensors and `feature_present_mask=0`.
- Missing required columns fail before training/inference.
- Null cells preserve `observed_mask=0` after imputation.
- No imputation crosses entity or split boundaries.
- Future values cannot affect past windows or preprocessing parameters.
- Alias collisions, invalid timestamps, duplicate-policy violations, and unit mismatches are
  reported deterministically.
- Reordering source columns does not change model inputs.

## Model tests

- All tensor shapes derive from registry, window, patch, mask, and target configuration.
- Frozen LLM parameters never receive updates.
- Checkpoint loading rejects incompatible registry/config digests.
- Tiny-data overfit test proves the pipeline can learn.
- Fixed seeds reproduce data splits and a tolerance-bounded smoke run.

## Evaluation

The exact primary metrics depend on D3 (task type). Required comparisons:

- naive persistence or prevalence baseline;
- simple non-LLM baseline;
- temporal encoder without semantic alignment;
- full frozen-LLM system;
- full system under increasing natural/simulated column missingness.

Report confidence intervals where feasible and avoid selecting models on the test set.

## Acceptance criteria

Before declaring the adaptation complete:

- All open decisions are resolved and recorded.
- Data contract and feature registry match an actual sample.
- Leakage, schema, mask, shape, and checkpoint-contract tests pass.
- Primary metric threshold is defined before final test evaluation.
- Performance does not collapse beyond an agreed tolerance for supported missing-column
  patterns.
- Unsupported schemas fail clearly or return an explicit degraded-status result.
- Reproduction instructions work from a clean environment without access to raw private
  data.
- A model card states intended use, unsupported use, training data scope, metrics,
  missingness behavior, and limitations.

## Definition of done for documentation-first milestone

This milestone is complete when this documentation is reviewed and the user supplies the
answers/sample needed for D1-D9. It intentionally does not claim that a working MIMC
forecaster exists yet.

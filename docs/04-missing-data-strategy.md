# Missing-data strategy

Missingness has three distinct meanings and must be modeled separately.

## 1. Absent column

The source has no column for a configured feature.

- Required feature: reject the input before model execution.
- Optional feature: materialize `NaN`, set `feature_present_mask=0`, and use the
  training-fitted fallback value only to produce a finite tensor.
- Always preserve presence information for the model and quality report.

## 2. Missing cell

The source contains the feature but a timestamp has no valid observation.

- Set `observed_mask=0`.
- Apply only the feature's configured causal imputer.
- Keep imputed values distinguishable through the mask and optional age channel.

## 3. Missing timestamp

No source row exists for an expected interval.

- Resample within an entity to the configured grid.
- Create a row with all dynamic values missing and masks set to 0.
- Never bridge forbidden gaps or entity/split boundaries.

## Baseline imputation hierarchy

For a numerical feature:

1. bounded forward fill using past values only;
2. optional domain-approved interpolation using past information only for online use;
3. training-set median as a finite fallback;
4. constant zero only after scaling when documented and accompanied by masks.

Backward fill is disallowed for forecasting because it leaks future measurements.
Categorical variables use an explicit unknown category fitted into the training
vocabulary.

## Scaling

- Fit scalers per feature on observed training values only.
- Do not include imputed validation/test values in fitted statistics.
- Prefer robust scaling for heavy-tailed measurements, configurable per feature.
- Persist scaler parameters and registry version with artifacts.

## Training robustness

Column dropout may simulate plausible absent optional features during training. Its
distribution must be configured from training data or a deployment scenario, never from
test outcomes. Random cell masking can be used similarly. Neither replaces evaluation
on naturally missing data.

## Monitoring and failure policy

Inference emits warnings or errors for:

- required columns absent;
- optional-column coverage below the model's documented support threshold;
- unseen aliases/units;
- excessive gap length or null rate;
- values outside configured plausible ranges;
- schema/checkpoint digest mismatch.

Predictions produced under degraded coverage carry the coverage report and must not be
indistinguishable from normal predictions.

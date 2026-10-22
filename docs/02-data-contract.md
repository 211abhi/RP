# Data contract

## Input model

The adapter accepts CSV or Parquet in long/wide tabular form. The first implementation
should normalize either form to one canonical wide table:

| Column class | Requirement | Description |
|---|---|---|
| `entity_id` | Required when multiple series exist | Stable, de-identified series key |
| `timestamp` | Required | Parseable event/measurement time |
| target column(s) | Required for training/evaluation | Configured prediction outcome |
| dynamic features | Optional unless configured required | Time-varying measurements |
| static features | Optional | Entity-level attributes joined after validation |

Physical names are mapped through configuration; the literal names above are logical
roles, not required source headers.

## Canonical feature registry

Every model build owns a versioned registry. Each feature entry defines:

```yaml
name: heart_rate
aliases: [hr, heartrate]
dtype: float32
unit: bpm
required: false
valid_range: [20, 260]
aggregation: median
imputation: forward_fill_then_train_median
max_forward_fill_steps: 2
include_value_mask: true
include_age_feature: true
```

The registry establishes feature order. Inputs are never alphabetically reordered and
never allowed to change tensor width.

## Canonical medication registry

Medication orders arrive as free text (`Vancomycin`, `vancomycin hcl`,
`Vancomycin Oral Liquid`), so a second versioned registry collapses them into canonical
classes:

```yaml
version: "mimic-drugs-v1"
drugs:
  - name: vancomycin
    match: ["vancomycin"]
  - name: beta_lactam
    match: ["piperacillin", "cefazolin", "meropenem", ...]
```

Rules:

1. Matching is lowercase substring, **first match wins**, so single-agent classes must
   precede grouped ones (`Heparin Flush` belongs to `heparin`).
2. No class's pattern may be a substring of another class's, or the earlier class silently
   swallows the later one. This is asserted by a test, not left to review.
3. Only `drug_type == MAIN` rows are read. Additive and base rows describe the diluent of a
   MAIN order and would double-count the same decision.
4. The registry establishes drug order and its digest is pinned into the checkpoint
   contract, exactly as the feature registry is.

Prescriptions are keyed by `hadm_id`, not by ICU stay, so the join runs through the
`hadm_id` carried on each stay and is then clipped to that stay's hour grid. Column naming
differs by release (`starttime`/`stoptime` in MIMIC-IV, `startdate`/`enddate` in MIMIC-III)
and is absorbed by aliasing.

An order with no stop time is bounded to `data.default_drug_hours`, never left open: an
unbounded interval would mark the class active for the rest of the admission and
manufacture positives the record does not support.

## Schema resolution

For each source:

1. Resolve configured ID, time, target, and feature aliases.
2. Reject ambiguous aliases (two source columns mapping to one canonical feature).
3. Reject missing required columns with an actionable error.
4. Materialize absent optional features as `NaN`.
5. Coerce types and units explicitly; failed coercions become validation errors or
   missing values according to configuration.
6. Sort within entity and apply the configured duplicate-time aggregation.
7. Resample only after train/validation/test assignment rules are established.
8. Emit a machine-readable schema report.

## Output tensors

For `F` canonical features and window length `T`, the adapter emits:

- `values`: float tensor `[N, T, F]` after train-fitted transformation;
- `observed_mask`: boolean/float tensor `[N, T, F]`, 1 only for an original valid value;
- `feature_present_mask`: tensor `[N, F]`, 1 if the source supplied that column;
- `time_since_observed`: optional float tensor `[N, T, F]`;
- `padding_mask`: tensor `[N, T]` when variable-length histories are supported;
- `targets`: task-dependent tensor;
- `drug_targets`: tensor `[N, D]` for `D` canonical drug classes, 1 if the class was active
  at any step of the prediction horizon;
- `drug_mask`: tensor `[N, D]`, 1 only where the source charts that class at all;
- metadata used for audit only, never blindly passed to the LLM.

Imputed values never flip an observation mask to 1.

A source with no prescriptions file yields `drug_mask = 0` for every class, which removes
those windows from the medication loss. It must never yield `drug_targets = 0` with
`drug_mask = 1`, because "not recorded" is not "not administered".

## Withheld-column contract

A column hidden at training time by `task.column_dropout_rate`, or withheld at inference by
`mimic-predict --features`, must be indistinguishable from a column the source never
recorded. All four input channels are set accordingly:

| channel | value for a withheld column |
|---|---|
| `values` | 0 after scaling, i.e. the training median |
| `observed_mask` | 0 |
| `feature_present_mask` | 0 |
| `time_since_observed` | maximum staleness |

Leaving any one channel intact would leak the value the imputation head is asked to
predict, and the resulting metric would be meaningless. This is asserted by a test.

## Data quality report

Each preparation run records:

- row/entity counts and time range;
- resolved, absent optional, and missing required columns;
- null rate and source-presence rate per feature;
- duplicates, invalid timestamps, coercion failures, and out-of-range counts;
- resampling coverage and window counts;
- registry version, config digest, split manifest, and train-fitted statistics ID.

## Leakage rules

- Split by entity when entities can recur; otherwise use strictly forward time splits.
- Fit normalization, vocabularies, imputers, thresholds, and labels on training data only.
- Never forward-fill across entity boundaries or split boundaries.
- Do not use measurements timestamped after the prediction cutoff.
- Targets and post-outcome proxies must not enter input features.
- Medication exposure is a target, never an input: the drug head predicts from vitals
  alone. Feeding current medications in would make a standing order trivially
  self-predicting.
- Medication positive weights are derived from training-split prevalence only.

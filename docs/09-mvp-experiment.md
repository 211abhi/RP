# MVP experiment specification

## Research question

From 24 hours of irregular, partially missing ICU vital signs, can one shared mask-aware
encoder support three predictions at once:

1. the next hour of every vital (forecast);
2. which medication classes are active during that hour (medication);
3. the current value of the vitals that were not supplied (imputation)?

This is an engineering validation experiment, not a clinical decision system.

## Cohort and unit of analysis

- Dataset: MIMIC-IV Clinical Database Demo v2.2.
- Entity: `stay_id`; leakage grouping: `subject_id`.
- Interval: from ICU `intime` through `outtime`, discretized to UTC-naive hourly bins.
- Aggregation: median of valid values within each stay/hour/feature.
- Minimum history: 24 hourly bins.

## Canonical features

| Feature | MIMIC-IV itemid | Valid range | Unit |
|---|---:|---:|---|
| heart rate | 220045 | 20–260 | bpm |
| respiratory rate | 220210 | 2–80 | breaths/min |
| SpO2 | 220277 | 40–100 | percent |
| systolic BP | 220179 | 30–300 | mmHg |
| diastolic BP | 220180 | 10–200 | mmHg |
| mean BP | 220181 | 20–250 | mmHg |
| temperature | 223761 | 80–115 | °F |

Out-of-range values are treated as unobserved and reported. Feature order comes only
from the registry.

## Forecast task

- Input: preceding 24 hourly bins.
- Target: all seven features in the next hourly bin.
- Windows whose target has no observed feature are omitted.
- Loss: mean squared error over observed target elements only.
- Primary metric: observed-target MAE in original units, macro-averaged across features.
- Secondary metrics: per-feature MAE/RMSE and coverage by missingness quartile.

## Medication task

- Input: the same 24 hourly bins of vitals. **Medications are never an input.**
- Target: for each of the canonical drug classes in `configs/drug_registry.yaml`, 1 if an
  order of that class was active at any hour of the prediction horizon.
- Exposure is taken from `prescriptions` start/stop intervals, joined on `hadm_id` and
  clipped to the stay's hour grid. An order with no stop time is bounded to
  `data.default_drug_hours`.
- Masking: a class contributes to the loss only where its source charts it at all. A source
  with no prescriptions file is excluded, never labelled all-negative.
- Loss: multi-label BCE over masked classes, with `pos_weight = (1 - p) / p` from
  training-split prevalence, capped at 20 so a one-in-thousands class cannot dominate.
- Primary metric: average precision per class, macro-averaged, against the
  predict-prevalence baseline.
- Secondary: per-class AUROC and the observed test prevalence beside it. Accuracy is not
  reported: standing ICU orders are long-lived and several classes exceed 80% prevalence,
  so accuracy would be high for a model with no signal.
- A class whose test split is all-positive or all-negative is reported as unscorable rather
  than given a fabricated number.

## Imputation task

- Input: the same window with a random `task.column_dropout_rate` fraction of the
  **present** columns hidden. No sample ever loses every column.
- A hidden column is zeroed in all four input channels and set to maximum staleness, so it
  is indistinguishable from a column the source never recorded.
- Target: the true scaled value of the hidden columns at the last context hour, scored only
  where that cell was genuinely observed.
- Loss: masked MSE. The head predicts a residual over the last context bin, so a supplied
  column round-trips for free.
- Primary metric: MAE in original units against training-median fill, which is what the
  pipeline would otherwise do for an absent column.
- The same dropout draw is used at evaluation under a fixed seed, so the number moves with
  the model and not with the random stream.

## Split contract

Subjects—not rows or stays—are deterministically assigned 70/15/15 to train/validation/
test using a seeded stable hash. A subject can occur in exactly one split. Preprocessing
statistics are fitted on observed training values only.

The 100-patient demo is too small for clinical conclusions. Its test split validates
software behavior only; the full-data experiment must repeat model selection without
reusing demo-test results as evidence.

## Preprocessing

1. Hourly median aggregation.
2. Preserve original observation masks.
3. Within each window, forward-fill each feature for at most six hours.
4. Fill remaining cells with the observed training median.
5. Robust-scale using training median and IQR; use scale 1 for zero-IQR features.
6. Concatenate scaled values, observation masks, feature-presence masks, and normalized
   time-since-observation channels at tokenizer input.

No backward fill or cross-stay fill is allowed.

## Baselines and ablations

Forecast:

- last observation carried forward;
- training median.

Medication:

- predict the training prevalence of each class (the AP floor);
- forecast-only encoder with a medication head trained alone, to show whether sharing the
  encoder helps or hurts.

Imputation:

- training-median fill;
- carry-forward from the column's own last observation, where one exists in the window.

Architecture:

- prediction heads directly on the time-series encoder;
- frozen LLM without semantic alignment;
- full model with alignment and soft prompt.

The LLM path is retained only if it improves the predefined validation metric over the
encoder-only baseline. Per-head loss weights are part of the ablation: a head is kept only
if it does not degrade the others beyond their reported confidence intervals.


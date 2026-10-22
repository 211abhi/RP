# Model card: MIMIC multi-task clinical predictor (development)

## Status

Trained multi-task model on the pooled MIMIC-III + MIMIC-IV demos (276 stays,
200 patients). Not validated for clinical use.

## Intended use

Research and software validation of masked multivariate modelling on deidentified ICU
data. One shared encoder over the preceding 24 hours of seven vital signs drives three
heads:

- **forecast** - the next hourly bin of every vital;
- **drug** - which of 36 canonical medication classes are active during that horizon,
  from vitals alone;
- **impute** - the current value of the vitals that were not supplied.

## Out-of-scope use

No diagnosis, triage, treatment recommendation, bedside alerting, patient monitoring,
or autonomous decision-making. Do not use results from the 100-patient demo as clinical
performance evidence.

**The medication head is explicitly not a prescribing aid.** It estimates which drug
classes are *associated* with a vital-sign trajectory in a historical single-centre
cohort. It has no access to diagnosis, allergy, renal function, interaction or dosing
information. Its labels say "an order was active", never "the order was correct", and
its training prevalences reflect one hospital's practice in a past decade.

**Imputed columns are model estimates, not measurements.** `mimic-predict` labels them as
such in every response. A downstream consumer that cannot distinguish an estimated vital
from a charted one must not be given them.

## Architecture

Mask-aware patch projection, temporal Transformer encoder, three task heads (forecast,
multi-label medication, column imputation), optional contrastive semantic alignment, and a
frozen causal language model with an optimized soft prompt. The forecast and imputation
heads predict a residual over the last observed context bin, so persistence is the
architectural floor. An encoder-only baseline is mandatory.

Training hides `task.column_dropout_rate` of each sample's supplied columns in all four
input channels - value, observed mask, present mask, staleness - which both supervises the
imputation head and stops the other two heads depending on a column some site will not
have. Medication labels carry a per-source presence mask, so a release with no
prescriptions file is excluded from that loss rather than trained towards all-negative.

## Results: forecast head

Two cohorts, one registry, one architecture, one set of weights. MIMIC-III and MIMIC-IV
reach the model through the same code path; release differences (itemid vocabulary,
column naming, quality flags) are absorbed in the registry and loader.

| Model | Cohort | Stays | Train windows | Macro MAE | Repeat-last-value | Skill (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| forecast-only | MIMIC-IV demo | 140 | 7,385 | 5.009 | 5.259 | +4.74% (+1.24 to +8.14) |
| forecast-only | MIMIC-III + IV pooled | 276 | 15,992 | 4.596 | 4.829 | +4.82% (+2.37 to +7.12) |
| **multi-task** | MIMIC-III + IV pooled | 276 | 15,992 | 4.612 | 4.829 | **+4.49% (+1.97 to +6.96)** |

Adding the medication and imputation heads did **not** cost forecast accuracy: +4.49% vs
+4.82%, with overlapping confidence intervals. The shared encoder carries three tasks for
the price of one.

Macro MAE is **not comparable across the two rows** — they are different test sets with
different intrinsic difficulty (note the differing persistence baselines). The comparable
quantity is skill over persistence, and it is flat at ~4.8%. What doubling the data bought
was not a better point estimate but a tighter one: the confidence interval narrowed from
6.9 to 4.8 percentage points wide. Both intervals exclude zero, so the model is reliably
better than repeating the last value, by a small margin.

### Forecast, pooled cohort, per vital (test split, 1,797 windows)

| Feature | MAE | RMSE | n |
|---|---:|---:|---:|
| heart_rate | 5.40 | 8.41 | 1,788 |
| respiratory_rate | 2.87 | 4.01 | 1,747 |
| spo2 | 1.28 | 2.33 | 1,732 |
| systolic_bp | 9.28 | 12.29 | 558 |
| diastolic_bp | 6.82 | 9.12 | 558 |
| mean_bp | 6.03 | 7.97 | 556 |
| temperature_f | 0.49 | 0.70 | 663 |

Rows are from the forecast-only model; the multi-task model's per-vital numbers are in
`artifacts/mimic-pooled/test_metrics.json`.

The head predicts a residual over the last observed context bin, so persistence is the
architectural floor rather than something the model must relearn. Blood pressure is
non-invasive cuff only; arterial-line itemids are deliberately unmapped, as are alarm
limits and ventilator settings, which pass a plain range check while not being patient
measurements. The alignment and frozen-LLM paths remain untrained.

These cohorts total 200 patients. The numbers validate software behaviour and justify
access to the full dataset; they are not clinical evidence.

## Results: medication head

36 canonical drug classes, pooled demo test split (1,797 windows), predicted from vitals
alone. Early stopping selected epoch 7; validation stopped improving after that while
training loss kept falling, which is overfitting on 200 patients.

| Metric | Value |
|---|---:|
| macro AUROC | 0.561 |
| macro average precision | 0.451 |
| predict-prevalence AP baseline | 0.405 |
| classes scored / unscorable | 36 / 0 |
| classes with AUROC > 0.6 | 14 / 36 |

**This is weak, and the weakness is the finding.** Macro AP clears the prevalence floor by
4.6 points, but more than half the classes sit at or below chance, and several are clearly
inverted (`aspirin` 0.187, `furosemide` 0.342, `nitroglycerin` 0.362). On a 200-patient
cohort an AUROC below 0.5 is not evidence of an inverse relationship; it is evidence the
test split is too small to estimate that class at all.

The classes that do carry signal are the ones with a physiological route to the vitals:

| Class | AUROC | AP | Test prevalence | Positives |
|---|---:|---:|---:|---:|
| acid_suppression | 0.793 | 0.940 | 0.798 | 1,434 |
| insulin | 0.780 | 0.987 | 0.954 | 1,714 |
| bronchodilator | 0.798 | 0.748 | 0.462 | 831 |
| vasopressor | 0.696 | 0.831 | 0.659 | 1,184 |
| warfarin | 0.827 | 0.228 | 0.090 | 162 |
| macrolide | 0.799 | 0.152 | 0.053 | 95 |

`gabapentinoid` reports AUROC 0.996 and a 156x AP lift on **2 test positives**. That is
noise with a decimal point, and it is listed here only so it is not mistaken for a result.

Standing ICU orders are long-lived (median order duration 31h), so class prevalences run
from 0.1% to 95%. Accuracy is therefore not reported: predicting "always on" would score
above 90% for `insulin` with no signal whatsoever. Average precision against prevalence is
the only honest headline.

## Results: imputation head

Withheld-column reconstruction at 30% column dropout, pooled demo test split, MAE in
original units against the training-median fill the pipeline would otherwise use.

| Vital | MAE | Median-fill MAE | Beats median fill |
|---|---:|---:|:--|
| mean_bp | 5.33 | 9.90 | yes, -46% |
| diastolic_bp | 7.33 | 12.25 | yes, -40% |
| systolic_bp | 10.81 | 14.07 | yes, -23% |
| temperature_f | 1.27 | 1.31 | marginal |
| respiratory_rate | 5.07 | 5.02 | **no** |
| spo2 | 2.12 | 1.97 | **no** |
| heart_rate | 16.81 | 16.22 | **no** |
| macro | 6.96 | 8.68 | yes, -19.8% |

The headline +19.8% is carried entirely by the three blood pressures, which are mutually
redundant: given two of them the third is nearly determined. Heart rate, respiratory rate
and SpO2 are **not** recoverable from the other vitals in this cohort, and the head does
slightly worse than the median for them. A consumer imputing those three should prefer the
median fill and know it.

This is the expected shape of the result, not a defect: the head can only exploit
correlations that exist. It does mean "predict any missing column" is too strong a claim —
the supported claim is "predict missing blood pressures".

## Known limitations

- Informative missingness can reflect care processes rather than physiology.
- Hourly aggregation discards within-hour dynamics.
- MIMIC represents a single academic medical center and historical care practices.
- Valid-range rules may remove unusual but genuine observations.
- The demo cohort is small and unsuitable for subgroup or safety conclusions.
- A frozen language model adds compute and may add no predictive value.
- Medication labels record that an order was active, not that it was administered, nor that
  it was clinically correct. Charting and ordering practice is itself a confounder.
- Medication prevalences reflect one academic centre's formulary and era; a model trained
  here would mispredict elsewhere even with identical physiology.
- Grouped drug classes (`vasopressor`, `opioid`, `beta_lactam`) hide within-class choice,
  which is often the decision that matters clinically.
- Imputation is only validated for the blood pressures; heart rate, respiratory rate and
  SpO2 are not recoverable from the remaining columns in this cohort.
- Column dropout is uniform at 30%. Real missingness is not uniform and not independent of
  patient state, so these imputation numbers are optimistic for correlated outages.
- The model overfits this cohort within ~7 epochs, so every number here is from an early
  checkpoint rather than a converged one.

## Required reporting

Report dataset release, cohort, split method, feature coverage, preprocessing state,
baselines, per-feature metrics, missingness-stratified metrics, and all ablations. Update
this card after every promoted checkpoint.


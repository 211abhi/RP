# Project status

Current milestone: multi-task predictor — medications, withheld columns, and vitals.

- [x] Preserve and map the original architecture.
- [x] Specify schema variability and missing-column behavior.
- [x] Specify missing-value, timestamp-gap, masking, and leakage policies.
- [x] Define configuration and checkpoint contracts.
- [x] Define implementation phases and acceptance criteria.
- [x] Confirm MIMIC-IV demo schema and canonical vital registry.
- [x] Freeze next-hour multivariate forecasting task and split contract.
- [x] Implement MIMIC-IV adapter, preprocessing, masks, windows, model, CLIs, and tests.
- [x] Pool MIMIC-III into the single loader via registry itemids (no per-release adapter).
- [ ] Extract and pool the MIMIC-II flat-file demo.
- [x] Add the medication-prediction head: prescriptions joined by hadm_id, free-text drug
      strings collapsed by `configs/drug_registry.yaml`, masked multi-label BCE with
      prevalence-derived positive weighting.
- [x] Add the column-imputation head: training-time column dropout hides a fraction of the
      supplied vitals in every input channel, and the head reconstructs them.
- [x] Serve a partial column set at inference via `mimic-predict --features`.
- [x] Use every visible GPU (`device: auto`, DataParallel, AMP, TF32) with no code change
      between CPU, one GPU, and many.
- [ ] Report medication AUROC/AP on a cohort large enough to resolve the rare classes; the
      demo test split has as few as 16 positives for some of them.
- [x] Run dependency-backed unit tests and train the encoder/head baseline.
- [ ] Approve alignment/LLM ablation before expensive soft-prompt optimization.
- [x] Train and evaluate the encoder/head baseline against persistence and median baselines.
- [x] Add `mimic-predict` for next-hour inference on raw stays.
- [x] Report skill over persistence with bootstrap confidence intervals.
- [ ] Apply for PhysioNet credentialed access to full MIMIC-IV.
- [ ] Train and evaluate the alignment/LLM path (gated on full data; demo test set is too small to resolve it).

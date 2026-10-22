# MIMIC multi-task clinical predictor

Mask-aware adaptation of `LLM-Adapted-Time-Series-Forecaster` for MIMIC ICU data whose
columns vary between files, sites, and inference requests.

Inputs: 7 vitals, 15 blood labs, and (optionally) the medication classes the patient is on
now. One shared encoder, four heads:

| head | question it answers | metric |
|---|---|---|
| **`drug_start`** | which medication classes will be **newly started within 24h** | AUROC / AP vs. prevalence |
| **`drug`** | which medication classes are active in the next hour | AP vs. carrying current meds forward |
| **`impute`** | the current value of the columns that were **not** supplied | MAE vs. training-median fill |
| **`forecast`** | the next hour of every vital and lab | MAE vs. repeat-last-value |

> Research prototype on the open MIMIC-III / MIMIC-IV demo cohorts (276 ICU stays,
> 200 patients). **Not for clinical use.**
>
> Pooled-demo test results: forecast **+4.5%** skill over repeat-last-value (95% CI +2.0 to
> +7.0); medication macro AP **0.451** vs. a 0.405 prevalence floor, with only 14 of 36
> classes above 0.6 AUROC; imputation **-19.8%** MAE vs. median fill, carried entirely by
> the blood pressures. The cohort is 200 patients and the model overfits it within ~7
> epochs. Read `MODEL_CARD.md` before quoting any of these.

## Architecture

```text
raw observations -> schema adapter -> masked patch tokenizer -> TS encoder --+
prescriptions ----> drug registry  -> hourly exposure labels                |
                                                                            v
                                            +------------- forecast head -> next-hour vitals
                                            +------------- drug head     -> P(class active)
                                            +------------- impute head   -> withheld columns

context/regime text --> frozen LLM embeddings -> alignment   (gated, see PROJECT_STATUS.md)
[soft prompt | encoded series + masks] -> frozen LLM -> head (gated)
```

The input boundary absorbs:

- optional feature columns, present at one site and absent at another;
- missing values inside present columns;
- irregular or duplicated timestamps;
- medication labels that one release has and another does not;
- train/inference schema differences, without changing tensor dimensions.

### Why the three heads share one encoder

All three ask about the same latent patient state, so they share its representation and
the small demo cohort supervises one encoder three times instead of three encoders once.
The `impute` head is also what makes the other two usable on a partial column set:
`task.column_dropout_rate` hides a fraction of the supplied columns on every training
sample, so the encoder never learns to depend on a column some site will not have.

A hidden column is zeroed in **all four** input channels — value, observed mask, present
mask, staleness — so it is indistinguishable from a column the source never recorded. That
is the whole reason `--features` works at inference.

### What the medication head is and is not

Labels come from `prescriptions` start/stop intervals. `drug` asks *"is this class active
in the next hour"*. Standing ICU orders are long (median 31h), so once current medications
are known, carrying them forward already scores AP ~0.98. The head therefore starts from
carry-forward (a learned per-class logit offset) and only learns stops; judge it against the
carry-forward line in the report, never against prevalence.

`drug_start` is the prescribing question: *"which class, off now, is started within
`task.start_horizon_hours`"*. Windows whose future runs past the end of the stay are
unscored rather than negative. This is the head to read.

Current medications are an input with their own known/unknown flag, and are hidden from
`task.medication_input_dropout` of training samples, so one model answers both "here is
what they are on" and "we don't know". The report scores every medication head both ways.
Per-class decision thresholds are fitted on validation (max F1) into `thresholds.json`,
which `mimic-predict` uses unless `--drug-threshold` overrides it.

Free-text drug strings are collapsed into canonical classes by
`configs/drug_registry.yaml`. Single-agent classes come first and matching is
first-match-wins, so `Heparin Flush` lands in `heparin`. Grouped classes
(`vasopressor`, `opioid`, `beta_lactam`, ...) are deliberate: a single agent with 13 orders
cannot be learned or evaluated on a demo cohort, but the clinical decision it stands for
can. Adding a release is a config entry plus registry patterns — never a new adapter module.

## Quick start (local)

```bash
python -m venv .venv
. .venv/bin/activate
# Install the right PyTorch build for this machine first when CUDA is required.
python -m pip install -e '.[dev]'

mimic-inspect  --config configs/mimic_pooled.yaml   # schema + label coverage
mimic-prepare  --config configs/mimic_pooled.yaml   # leakage-safe windows + drug labels
mimic-train    --config configs/mimic_pooled.yaml   # all three heads
mimic-evaluate --config configs/mimic_pooled.yaml   # three-section test report
```

`configs/mimic_pooled.yaml` pools every release under `data.sources` into one model.
Subject ids are namespaced per source, so no patient can straddle a train/test boundary.

### Predicting from a partial column set

Name only the columns the site actually charts. Everything else is treated as absent and
returned under `imputed_columns`, flagged as estimated:

```bash
mimic-predict --config configs/mimic_pooled.yaml \
              --stay-id 30057454 \
              --features heart_rate,spo2,respiratory_rate
```

```json
{
  "supplied_columns": ["heart_rate", "respiratory_rate", "spo2"],
  "withheld_columns": ["systolic_bp", "diastolic_bp", "mean_bp", "temperature_f"],
  "note": "imputed_columns are model estimates, not measurements",
  "predictions": [{
    "forecast": [{"hour": "...", "values": {"heart_rate": 108.7, "...": "..."}}],
    "imputed_columns": {"systolic_bp": 103.2, "diastolic_bp": 63.5, "mean_bp": 75.1,
                        "temperature_f": 99.3},
    "medications": {"predicted": ["opioid", "acid_suppression", "..."],
                    "probability": {"insulin": 0.91, "...": "..."}}
  }]
}
```

For your own patient instead of a MIMIC stay, pass a CSV with a `time` column and any
subset of the vitals, covering at least 24 hours (format: `examples/patient.csv`). Columns
the file lacks are imputed:

```bash
mimic-predict --config configs/mimic_pooled.yaml --patient-csv examples/patient.csv
```

Lab columns (`glucose, potassium, sodium, bicarbonate, bun, creatinine, magnesium, calcium,
wbc, hemoglobin, platelets, inr, bilirubin, lactate, ph`) go in the same CSV, blank between
draws; each is held for up to 24h. Add `--current-meds heparin,insulin` (or `none`) when the
current medication classes are known. The response then carries `medications` (active next
hour) and `new_starts` (likely started within 24h):

```bash
mimic-predict --config configs/runtime_best.yaml --patient-csv examples/patient.csv \
              --current-meds heparin,acid_suppression
```

Omit `--features` to supply every column; `imputed_columns` is then empty. `mimic-predict`
refuses any checkpoint whose config or registry digest differs from the current one.

## Training in a notebook — Kaggle, Colab, or Google Cloud

`run.ipynb` is the one-file trainer. It detects the runtime, **uses every GPU it can see**,
downloads the open demos if they are absent, trains, evaluates, and zips the artifacts.
Set `TRAIN_EPOCHS` interactively or as an environment variable.

### Kaggle

1. **New Notebook** → **File ▸ Import Notebook** → upload `run.ipynb`.
2. **Settings ▸ Accelerator** → **GPU T4 x2** (both are used; see *How all GPUs are used*).
3. **Settings ▸ Internet** → **On**. The notebook downloads the demo cohorts from
   PhysioNet, and `pip install -e .` needs it.
4. Get the code onto the machine, either:
   - **Add Data ▸ Upload** a zip of this repo (it lands read-only under `/kaggle/input/…`;
     copy it to `/kaggle/working/MIMC-LLM-Time-Series-Forecaster` first, because the
     notebook writes artifacts next to the project), **or**
   - set `PROJECT_REPO_URL` in cell 2's environment and let it `git clone`.
5. **Run All**. A 60-epoch pooled-demo run takes a few minutes on 2×T4.
6. Collect `mimic_multitask_artifacts.zip` from the **Output** tab, or
   **Save Version ▸ Save & Run All** to keep it after the session.

Kaggle sessions stop at 12h (9h for GPU) and `/kaggle/working` is wiped, so always take the
zip. For an unattended run use **Save & Run All (Commit)**; the notebook detects batch mode
and takes the default epoch count without prompting.

### Google Colab

1. **File ▸ Upload notebook** → `run.ipynb`.
2. **Runtime ▸ Change runtime type** → any GPU (A100/L4/T4). Colab gives one GPU; the code
   uses more when more are visible.
3. Either mount Drive and put the repo there, or set `PROJECT_REPO_URL`:
   ```python
   import os; os.environ['PROJECT_REPO_URL'] = 'https://github.com/<you>/<repo>.git'
   ```
4. **Runtime ▸ Run all**, then download the zip from the file browser.

### Google Cloud

**Option A — Vertex AI Workbench (notebook, closest to Kaggle/Colab).**

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable notebooks.googleapis.com aiplatform.googleapis.com

# One instance, four GPUs. All four are used.
gcloud notebooks instances create mimic-trainer \
  --location=us-central1-a \
  --machine-type=n1-standard-16 \
  --accelerator-type=NVIDIA_TESLA_T4 --accelerator-core-count=4 \
  --vm-image-project=deeplearning-platform-release \
  --vm-image-family=pytorch-latest-gpu \
  --install-gpu-driver \
  --boot-disk-size=200GB
```

Open JupyterLab from the Vertex AI ▸ Workbench console, upload `run.ipynb`, **Run All**.
**Stop the instance when finished — a running GPU VM bills by the second.**

```bash
gcloud notebooks instances stop   mimic-trainer --location=us-central1-a
gcloud notebooks instances delete mimic-trainer --location=us-central1-a   # when done
```

**Option B — Vertex AI custom training job (unattended, no notebook).** The CLIs are the
container entrypoint, so no notebook is needed:

```bash
BUCKET=gs://your-bucket
cat > Dockerfile <<'EOF'
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-4.py310
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e '.[dev]'
ENTRYPOINT ["bash","-lc","\
  mimic-prepare  --config configs/mimic_pooled.yaml && \
  mimic-train    --config configs/mimic_pooled.yaml && \
  mimic-evaluate --config configs/mimic_pooled.yaml && \
  gsutil -m cp -r artifacts/mimic-pooled $BUCKET/runs/$(date +%Y%m%d-%H%M%S)/"]
EOF

REGION=us-central1
IMAGE=$REGION-docker.pkg.dev/YOUR_PROJECT_ID/mimic/trainer:latest
gcloud artifacts repositories create mimic --repository-format=docker --location=$REGION
gcloud builds submit --tag $IMAGE .

gcloud ai custom-jobs create \
  --region=$REGION --display-name=mimic-multitask \
  --worker-pool-spec=machine-type=n1-standard-16,replica-count=1,\
accelerator-type=NVIDIA_TESLA_T4,accelerator-count=4,container-image-uri=$IMAGE
```

The image must carry `data/raw/` (or fetch it in the entrypoint) — the demo cohorts are
open access, but full MIMIC-IV requires credentialed PhysioNet access and must never be
baked into a shared image.

### How all GPUs are used

`project.device: auto` takes CUDA when present, then MPS, then CPU. With more than one
visible CUDA device the model is wrapped in `nn.DataParallel`, which splits each batch
across all of them, and the notebook scales `batch_size` by the device count so the
per-GPU batch stays constant. Mixed precision (`training.amp`) and TF32 are enabled on
CUDA and ignored elsewhere. Nothing needs changing between a 1-GPU Colab box, Kaggle's
2×T4, and a 4-GPU Vertex instance.

`DataParallel` rather than `DistributedDataParallel` on purpose: DDP needs a process
launcher that a notebook kernel has no clean way to provide. For these device counts the
difference is throughput, not correctness. For an 8-GPU-plus run, switch to
`torchrun`-launched DDP — the batch-level split is the bottleneck by then.

## Configuration

Key knobs beyond the originals, all in `configs/*.yaml`:

| key | meaning |
|---|---|
| `data.drug_registry` | canonical medication classes and their name patterns |
| `data.default_drug_hours` | how long an order with no stop time counts as active (not forever) |
| `task.column_dropout_rate` | fraction of supplied columns hidden per sample; supervises `impute` |
| `task.loss_weights` | per-head weights (`forecast`, `drug`, `drug_start`, `impute`); `0` switches a head off |
| `task.medication_inputs` | feed current medication classes as an input |
| `task.medication_input_dropout` | fraction of training samples with medications hidden (unknown) |
| `task.start_horizon_hours` | window for the new-start label |
| feature `table` / `max_fill_hours` | `labevents` for labs; per-feature forward-fill limit |
| `project.device` | `auto`, or pin `cuda` / `cuda:1` / `cpu` / `mps` |
| `training.amp` | mixed precision; takes effect on CUDA only |

Set `loss_weights: {forecast: 1.0, drug: 0.0, impute: 0.0}` to reproduce the original
forecast-only model. `column_dropout_rate` must be positive whenever `impute` is weighted —
the config refuses the combination otherwise, because the head would have no supervision.

## Choosing the best settings

`mimic-sweep --config <cfg> --trials 16` trains random settings (width, depth, patch size,
dropout, learning rate, weight decay) on the same prepared windows and keeps the lowest
validation loss; trial 0 is the config unchanged, and the test split is never consulted.
It writes `configs/<cfg>_best.yaml` pointing at the winner's checkpoint. The notebook
runs it by default (`SWEEP_TRIALS`).

## Full MIMIC-IV

The demo cohort is the ceiling on accuracy. With credentialed PhysioNet access, download
MIMIC-IV 3.1 to `data/raw/mimic-iv-3.1/` and use `configs/mimic_iv_full.yaml` (12h window
stride, so the windows fit in memory). On Kaggle, upload it as a **private** dataset and
attach it instead of the demo: the notebook detects it and switches stride on its own. Start
with `LIMIT_STAYS=20000` to check the RAM headroom; the loader is pure Python and has not
been run against the full release.

## Documentation

1. [Requirements and assumptions](docs/01-requirements-and-assumptions.md)
2. [Data contract](docs/02-data-contract.md)
3. [Architecture](docs/03-architecture.md)
4. [Missing-data strategy](docs/04-missing-data-strategy.md)
5. [Configuration reference](docs/05-configuration.md)
6. [Implementation plan](docs/06-implementation-plan.md)
7. [Validation and acceptance](docs/07-validation-and-acceptance.md)
8. [MVP experiment contract](docs/09-mvp-experiment.md)
9. [Operations](docs/10-operations.md)

`MODEL_CARD.md` records intended use and limitations; `PROJECT_STATUS.md` tracks the
milestone list. Run `pytest` for the data-contract and head-behaviour checks.

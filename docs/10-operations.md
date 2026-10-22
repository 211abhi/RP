# Operations and reproducibility

## Environment

Python 3.10+ is required. Install PyTorch separately for the machine's CUDA version,
then install this project:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,llm]'
```

The data-only inspection and preparation code has no pandas dependency. Raw data stays
under `data/`, which is ignored by Git.

## Pipeline

```bash
mimic-inspect --config configs/mimic_iv_demo.yaml
mimic-prepare --config configs/mimic_iv_demo.yaml
mimic-train --config configs/mimic_iv_demo.yaml --stage encoder
mimic-train --config configs/mimic_iv_demo.yaml --stage head
mimic-evaluate --config configs/mimic_iv_demo.yaml
```

Soft-prompt optimization is an explicit later stage because it is expensive:

```bash
mimic-train --config configs/mimic_iv_demo.yaml --stage soft-prompt
```

## Artifacts

Each run writes to `artifacts/<run_id>/`:

- resolved configuration;
- data-quality and split reports;
- feature registry digest and preprocessing state;
- prepared arrays or manifest;
- checkpoints with model/data contract metadata;
- metrics and environment metadata.

Never put raw patient-level rows, identifiers, or free-form text in logs or model
prompts. Subject/stay identifiers may exist in local split manifests but are not copied
into checkpoints.

## Resume and failure behavior

- Downloads use `wget -N -c` and can resume.
- Preparation is atomic: write to a temporary output, validate, then replace.
- Training checkpoints include epoch, optimizer state, config digest, and feature digest.
- Incompatible digests fail closed.
- Empty splits, zero observed targets, non-finite loss, or overlapping subjects stop the
  run with a non-zero exit status.

## Full-data transition

The demo and full MIMIC-IV releases share schema, but the transition still requires:

1. credentialed access and compliance with the PhysioNet data-use agreement;
2. a new immutable raw-data path;
3. regeneration of splits and preprocessing state;
4. capacity estimates for `chartevents` streaming;
5. selection on full-data validation only;
6. an updated model card and dataset manifest.


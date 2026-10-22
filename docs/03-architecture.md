# Architecture

## Preserved architecture

The new project retains the original five learned stages:

1. A patch tokenizer converts a history window into patch tokens.
2. A temporal encoder (1-D convolution plus Transformer) produces a sequence/CLS
   representation.
3. Contrastive training aligns time-series representations with frozen LLM embeddings
   of domain-relevant context labels.
4. CMA-ES optimizes a soft prompt in a low-dimensional random subspace while the LLM
   remains frozen.
5. A small prediction head maps frozen LLM state to the configured output.

## New input boundary

```text
source files
  -> schema validator / alias resolver
  -> entity-time canonicalizer
  -> split manifest
  -> train-fitted resampling, imputation and scaling
  -> values + observation masks + column-presence masks + time deltas
  -> masked patch tokenizer
```

## Mask-aware tokenizer

The old tokenizer assumes `patch_dim = patch_size * n_features`. The replacement uses a
fixed canonical `F` and one of two configurable fusion methods:

- `concat` (baseline): concatenate values, observed masks, feature-presence masks
  broadcast over time, and optional time deltas before patch projection.
- `gated` (preferred after baseline): independently project values and missingness,
  then learn a gate. This distinguishes a real zero from an imputed zero.

The tokenizer validates its calculated input dimension at startup; configuration cannot
silently disagree with the registry.

## Semantic alignment

Finance labels (`TRENDING_UP`, `BREAKOUT`, and so on) must not be reused. A domain label
provider should produce only labels justified by pre-cutoff input data. Options are:

- unsupervised cluster/state descriptions fitted on training data;
- task-specific state labels supplied by domain rules;
- simple missingness/trajectory descriptions as a safe baseline.

Templates are versioned and contain no raw identifiers or free-form sensitive records.
If reliable semantic labels are unavailable, the alignment phase becomes optional and
must be evaluated through an ablation rather than filled with arbitrary labels.

## Prediction head

The target decision selects the head and objective:

| Task | Output | Typical objective |
|---|---|---|
| continuous forecast | `[batch, horizon, targets]` | MAE/MSE/Huber |
| binary event | `[batch, horizon or 1]` logits | BCE with logits |
| multiclass state | `[batch, classes]` logits | cross entropy |
| time-to-event | task-specific risk/survival output | approved survival loss |

The first implementation should support one explicitly selected task, not a speculative
universal head.

## Checkpoint contract

Every checkpoint includes or references:

- canonical registry version and ordered feature list;
- preprocessing/config digest;
- tokenizer input layout and dimensions;
- target specification and horizon;
- training split manifest ID;
- model version and source commit.

Loading fails closed on incompatible contracts unless an explicit migration exists.

## Proposed layout

```text
configs/
  base_config.example.yaml
  feature_registry.example.yaml
docs/
src/
  data/          schema, validation, canonicalization, splits, preprocessing
  tokenizer/     mask-aware patch tokenizer
  encoder/       temporal encoder and alignment trainer
  alignment/     label provider, templates, contrastive loss
  llm/           frozen LLM wrapper
  soft_prompt/   CMA-ES soft prompt
  prediction/    task-specific heads and forecaster
  evaluation/    metrics, subgroup/missingness analyses
  utils/         reproducibility, checkpoints, reports
scripts/         numbered pipeline entry points
tests/           unit, integration, leakage, and schema-drift tests
artifacts/       ignored generated manifests/reports/checkpoints
```

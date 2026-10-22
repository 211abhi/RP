# Configuration reference

Configuration is the source of truth; derived tensor dimensions are calculated and
validated by code.

```yaml
project:
  name: mimc-llm-time-series-forecaster
  seed: 42

data:
  format: parquet                 # csv | parquet
  paths: []
  entity_id_col: null             # required for multi-entity data
  timestamp_col: null             # must be confirmed
  timezone: UTC
  frequency: null                 # e.g. 1h; must be confirmed
  duplicate_policy: aggregate     # error | first | last | aggregate
  feature_registry: configs/feature_registry.yaml
  split:
    strategy: grouped_temporal    # grouped_temporal | temporal
    train: 0.70
    val: 0.15
    test: 0.15

target:
  task: null                      # regression | binary | multiclass | survival
  columns: []
  horizon_steps: null

preprocessing:
  resample: true
  emit_observed_mask: true
  emit_feature_present_mask: true
  emit_time_since_observed: true
  fail_on_missing_required: true
  max_optional_missing_fraction: 1.0

tokenizer:
  window_size: null
  patch_size: 16
  fusion: concat                  # concat | gated
  train_stride: 1
  eval_stride: 8

encoder:
  embed_dim: 256
  n_heads: 8
  n_layers: 4
  ffn_dim: 512
  dropout: 0.1
  output_dim: 256

alignment:
  enabled: true
  label_provider: null            # must be selected and documented
  temperature: 0.07
  projection_dim: 1024

llm:
  model_name: gpt2-medium
  hidden_dim: 1024
  freeze_all: true

soft_prompt:
  n_tokens: 32
  token_dim: 1024

cmaes:
  subspace_dim: 500
  popsize: 20
  sigma0: 0.1
  maxiter: 100

prediction_head:
  hidden_dim: 256
  dropout: 0.1

evaluation:
  metrics: []                     # chosen from task type
  report_by_missingness: true
  report_by_entity: false         # enable only when privacy-safe
```

## Startup invariants

- `window_size` is divisible by `patch_size`, unless padding is explicitly enabled.
- Registry feature names and aliases are unique.
- Target columns cannot also be input features unless explicitly lagged.
- LLM, projection, soft-prompt, and prediction-head dimensions agree.
- Split ratios sum to 1 and no entity/time leakage is detected.
- Checkpoint contract matches registry and preprocessing digests.

## Implemented keys

The block above is the original design sketch and is broader than what ships. The
authoritative shapes are `configs/mimic_pooled.yaml` (pooled, multi-source) and
`configs/mimic_iv_demo.yaml` (single source), validated by
`mimic_forecaster.config.validate_config`. The multi-task additions are:

```yaml
data:
  drug_registry: configs/drug_registry.yaml   # canonical medication classes
  default_drug_hours: 24                      # an order with no stop time is bounded, not open
  sources:
    - name: mimic-iv-demo
      prescriptions: hosp/prescriptions.csv.gz  # optional; absent -> labels masked out

task:
  type: multitask
  column_dropout_rate: 0.3    # supplied columns hidden per sample; supervises the impute head
  loss_weights:               # a head set to 0 is switched off
    forecast: 1.0
    drug: 0.5
    impute: 0.5

project:
  device: auto                # every visible CUDA GPU, else MPS, else CPU

training:
  amp: true                   # mixed precision; takes effect on CUDA only
```

Additional startup invariants enforced for these keys:

- `column_dropout_rate` is in `[0, 1)`.
- `loss_weights` names only `forecast`, `drug`, `impute`; all are non-negative and at
  least one is positive.
- A positive `impute` weight requires `column_dropout_rate > 0`, or the head would have
  no supervision.
- No drug-registry match pattern may be a substring of another class's pattern, or
  first-match-wins would silently swallow the later class.

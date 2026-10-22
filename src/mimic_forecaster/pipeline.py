"""Shared model construction."""

from __future__ import annotations

from typing import Any

from .models import MaskAwarePatchTokenizer, MultiTaskForecaster, TimeSeriesEncoder


def build_forecaster(cfg: dict[str, Any], n_features: int, n_drugs: int = 0):
    context = int(cfg["task"]["context_hours"])
    patch = int(cfg["tokenizer"]["patch_size"])
    embed = int(cfg["tokenizer"]["embed_dim"])
    drug_inputs = n_drugs if cfg["task"].get("medication_inputs", False) else 0
    tokenizer = MaskAwarePatchTokenizer(n_features, patch, embed, include_age=bool(cfg["tokenizer"].get("include_age", True)),
                                        n_drug_inputs=drug_inputs)
    encoder = TimeSeriesEncoder(embed, int(cfg["encoder"]["n_heads"]), int(cfg["encoder"]["n_layers"]),
                                int(cfg["encoder"]["ffn_dim"]), float(cfg["encoder"]["dropout"]), context // patch)
    return MultiTaskForecaster(tokenizer, encoder, embed, int(cfg["task"]["horizon_hours"]), n_features,
                               n_drugs, float(cfg["encoder"]["dropout"]))

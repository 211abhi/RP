"""Multi-task encoder baseline and frozen-LLM forecaster.

Three heads share one mask-aware encoder, because all three answer questions about the
same latent patient state:

  forecast   next `horizon` hours of every vital          (regression, masked MSE)
  drug       which medication classes are active next     (multi-label, masked BCE)
  start      which classes are newly started within a day (multi-label, masked BCE)
  impute     the current value of the columns NOT given   (regression, masked MSE)

The impute head is what makes a partial column set usable at inference: feed whichever
vitals a site actually charts, and it reconstructs the rest from the observed ones.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoder import TimeSeriesEncoder
from .frozen_llm import FrozenLLM
from .soft_prompt import SoftPrompt
from .tokenizer import MaskAwarePatchTokenizer


class MultiTaskForecaster(nn.Module):
    def __init__(self, tokenizer: MaskAwarePatchTokenizer, encoder: TimeSeriesEncoder,
                 embed_dim: int, horizon: int, n_targets: int, n_drugs: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__(); self.tokenizer = tokenizer; self.encoder = encoder
        self.horizon = horizon; self.n_targets = n_targets; self.n_drugs = n_drugs
        # `head` keeps its original name so forecast-only checkpoints stay loadable.
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, horizon * n_targets))
        self.impute_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, n_targets))
        self.drug_head = (nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, n_drugs))
                          if n_drugs else None)
        self.start_head = (nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, n_drugs))
                           if n_drugs else None)
        # Per-class logit offset for "on it at the last hour". Orders run for about a day, so
        # carrying current medications forward is a very strong prior; starting from it means
        # the head only has to learn stops. Inert when medications are unknown.
        self.carry = nn.Parameter(torch.full((n_drugs,), 3.0)) if n_drugs else None

    def forward(self, values, observed, present, age, drug_history=None, drug_known=None) -> dict[str, torch.Tensor]:
        state = self.encoder(self.tokenizer(values, observed, present, age, drug_history, drug_known))
        last = values[:, -1, :]
        out = {
            # Residual on the last context bin, i.e. learn the delta from persistence.
            "forecast": self.head(state).reshape(len(values), self.horizon, self.n_targets) + last.unsqueeze(1),
            # Residual on the last bin too, so a present column round-trips for free and
            # the head spends its capacity on the columns that are actually absent.
            "impute": self.impute_head(state) + last,
        }
        if self.drug_head is not None:
            out["drug_logits"] = self.drug_head(state)
            if drug_history is not None and drug_history.shape[-1] == self.n_drugs:
                out["drug_logits"] = out["drug_logits"] + self.carry * (2.0 * drug_history[:, -1, :] - 1.0) * drug_known
            out["start_logits"] = self.start_head(state)
        return out


class FrozenLLMForecaster(nn.Module):
    def __init__(self, tokenizer: MaskAwarePatchTokenizer, encoder: TimeSeriesEncoder, llm: FrozenLLM,
                 prompt: SoftPrompt, encoder_dim: int, horizon: int, n_targets: int) -> None:
        super().__init__(); self.tokenizer = tokenizer; self.encoder = encoder; self.llm = llm; self.prompt = prompt
        self.bridge = nn.Linear(encoder_dim, llm.hidden_dim)
        self.head = nn.Sequential(nn.Linear(llm.hidden_dim, 256), nn.GELU(), nn.Linear(256, horizon * n_targets))
        self.horizon = horizon; self.n_targets = n_targets

    def forward(self, values, observed, present, age):
        state = self.bridge(self.encoder(self.tokenizer(values, observed, present, age))).unsqueeze(1)
        inputs = torch.cat([self.prompt(len(values)), state], dim=1)
        return self.head(self.llm(inputs)).reshape(len(values), self.horizon, self.n_targets)

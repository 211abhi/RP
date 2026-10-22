"""Frozen Hugging Face causal-LM wrapper loaded only when enabled."""

from __future__ import annotations

import torch
from torch import nn


class FrozenLLM(nn.Module):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("install the 'llm' optional dependency to enable the frozen LLM") from error
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_dim = int(self.model.config.hidden_size)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        result = self.model(inputs_embeds=embeddings, output_hidden_states=True, use_cache=False)
        return result.hidden_states[-1][:, 0]


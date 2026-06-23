"""DAPN domain translator G (Zhao et al., arXiv:2003.08626)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainTranslator(nn.Module):
    """MLP translator with optional residual connection when dims match."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = input_dim == output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.residual else y


    def load_state_dict(self, state, strict=True):
        clean = {k: v for k, v in state.items() if k.startswith("net.")}
        return super().load_state_dict(clean, strict=strict)


def load_translator(path: str, device: str = "cpu") -> DomainTranslator:
    ckpt = torch.load(path, map_location=device)
    if "net.0.weight" in ckpt:
        inp = int(ckpt["input_dim"])
        out = int(ckpt["output_dim"])
        state = ckpt
    else:
        inp = ckpt.get("input_dim", 26)
        out = ckpt.get("output_dim", 26)
        state = ckpt.get("state_dict", ckpt)
    model = DomainTranslator(inp, out)
    model.load_state_dict(state)
    model.eval()
    return model

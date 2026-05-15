"""Gate modules used by DISeL.

`RankGate` is a per-rank input-dependent sigmoid mask that multiplies the LoRA
A-projection output. `LightRankGate` factors the gate's projection through a
bottleneck for parameter efficiency.

Both modules return a tensor of shape (..., rank) with values in (0, 1).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RankGate(nn.Module):
    r"""Input-dependent per-rank gate: :math:`g(x) = \sigma(W_g x + b_g)`.

    Args:
        in_features: Input dimension (matches the LoRA A-projection input).
        rank: LoRA rank (size of the gate's output).
        bias_init: Initial bias for the linear projection. Default ``-3.0``
            makes the gate start nearly closed (sigmoid(-3) ≈ 0.047).
        normalize: If True, scale the pre-sigmoid logits by
            :math:`1/\sqrt{d_\text{in}}` to stabilise large hidden dims.
        weight_init: ``"random"`` keeps the default Kaiming-uniform init
            of :class:`nn.Linear`; ``"zero"`` zeroes :math:`W_g` so the gate
            starts as a constant :math:`\sigma(b_g)`.
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        bias_init: float = -3.0,
        normalize: bool = False,
        weight_init: str = "random",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, rank)
        self.normalize = normalize
        if normalize:
            self.register_buffer(
                "scale",
                torch.tensor(1.0 / math.sqrt(in_features)),
                persistent=False,
            )

        if weight_init == "zero":
            nn.init.zeros_(self.linear.weight)
        elif weight_init != "random":
            raise ValueError(
                f"weight_init must be 'random' or 'zero', got {weight_init!r}"
            )
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if self.normalize:
            h = h * self.scale
        return torch.sigmoid(h)


class LightRankGate(nn.Module):
    r"""Low-rank bottleneck variant: :math:`g(x) = \sigma(U V x + b_g)`.

    Reduces the gate parameter count from :math:`r \cdot d_\text{in}` to
    :math:`(d_\text{in} + r) \cdot d_b`.

    Args:
        in_features: Input dimension.
        rank: LoRA rank.
        bottleneck_dim: Bottleneck size (typical: 8–32).
        bias_init: Initial bias on the up-projection. Default ``-3.0``.
        normalize: As in :class:`RankGate`.
        weight_init: ``"random"`` (Kaiming-uniform on both projections) or
            ``"zero"`` (zeros the up-projection so the gate starts as
            :math:`\sigma(b_g)`).
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        bottleneck_dim: int = 16,
        bias_init: float = -3.0,
        normalize: bool = False,
        weight_init: str = "random",
    ) -> None:
        super().__init__()
        self.proj_down = nn.Linear(in_features, bottleneck_dim, bias=False)
        self.proj_up = nn.Linear(bottleneck_dim, rank)
        self.normalize = normalize
        if normalize:
            self.register_buffer(
                "scale",
                torch.tensor(1.0 / math.sqrt(in_features)),
                persistent=False,
            )

        if weight_init == "zero":
            nn.init.zeros_(self.proj_up.weight)
        elif weight_init != "random":
            raise ValueError(
                f"weight_init must be 'random' or 'zero', got {weight_init!r}"
            )
        nn.init.constant_(self.proj_up.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_up(self.proj_down(x))
        if self.normalize:
            h = h * self.scale
        return torch.sigmoid(h)

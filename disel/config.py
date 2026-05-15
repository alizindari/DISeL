"""Configuration object for DISeL.

`DiselConfig` extends `peft.LoraConfig` with five extra fields. Constructing it
and passing the resulting object to `peft.get_peft_model` produces a regular
PEFT-LoRA model; the DISeL gates are attached by a separate call to
:func:`disel.enable_disel`, which mirrors PEFT's variant-registration logic
without monkey-patching internal methods.

This separation exists so that the project can be released as a standalone
`pip install disel` package today, and migrated to a `use_disel=True` flag on
`LoraConfig` upstream of PEFT later with a minimal diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from peft import LoraConfig


@dataclass
class DiselConfig(LoraConfig):
    """LoRA config with extra fields for DISeL gates.

    Args:
        disel_gate_bias_init: Initial bias for the gate (default ``-3.0``).
        disel_gate_normalize: Apply ``1/sqrt(d_in)`` scaling before sigmoid.
        disel_gate_weight_init: ``"random"`` (Kaiming) or ``"zero"``.
        disel_gate_bottleneck_dim: If ``None`` (default), use the full
            :class:`disel.layer.RankGate`. If an int, use
            :class:`disel.layer.LightRankGate` with this bottleneck size.
        disel_gate_lr_multiplier: Multiplier applied on top of the base learning
            rate to obtain the gate learning rate. Consumed by
            :func:`disel.build_optimizer`; PEFT itself does not own the
            optimizer, so this field is informational.
    """

    disel_gate_bias_init: float = field(default=-3.0)
    disel_gate_normalize: bool = field(default=False)
    disel_gate_weight_init: Literal["random", "zero"] = field(default="random")
    disel_gate_bottleneck_dim: int | None = field(default=None)
    disel_gate_lr_multiplier: float = field(default=5.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.disel_gate_bottleneck_dim is not None and self.disel_gate_bottleneck_dim < 1:
            raise ValueError(
                "disel_gate_bottleneck_dim must be a positive integer or None, "
                f"got {self.disel_gate_bottleneck_dim}"
            )
        if self.disel_gate_weight_init not in ("random", "zero"):
            raise ValueError(
                "disel_gate_weight_init must be 'random' or 'zero', "
                f"got {self.disel_gate_weight_init!r}"
            )

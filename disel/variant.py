"""DISeL as a :class:`peft.tuners.lora.layer.LoraVariant`.

The class shape matches PEFT's internal variant pattern (e.g. ``DoraLinearVariant``)
exactly, so upstreaming this file into ``src/peft/tuners/lora/variants.py`` is a
near-direct copy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from peft.tuners.lora.layer import LoraLayer, LoraVariant


class DiselLinearVariant(LoraVariant):
    """Input-dependent per-rank gate on the LoRA branch.

    Forward computation (per active adapter):

    .. math::
        h = \\text{lora\\_B}\\bigl(g(x) \\odot \\text{lora\\_A}(\\text{dropout}(x))\\bigr)
            \\cdot \\text{scaling}

    where :math:`g(x) = \\sigma(W_g x + b_g) \\in (0, 1)^r` is produced by
    ``module.disel_gate[active_adapter]``.

    The gate is input-dependent, so ``merge_and_unload`` is not supported.
    """

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        # Gate construction and registration is handled by
        # `disel.enable_disel(...)`. We add the `disel_gate` ModuleDict name to
        # the layer's `adapter_layer_names` here so PEFT's save/load picks it
        # up alongside the LoRA matrices.
        if "disel_gate" not in module.adapter_layer_names:
            module.adapter_layer_names = module.adapter_layer_names + ("disel_gate",)

    @staticmethod
    def merge_safe(module, active_adapter, orig_weight):  # noqa: D401
        raise NotImplementedError(
            "DISeL gates are input-dependent and cannot be folded into a "
            "static weight matrix; merge_and_unload is not supported."
        )

    @staticmethod
    def merge_unsafe(module, active_adapter, orig_weight):
        raise NotImplementedError(
            "DISeL gates are input-dependent and cannot be folded into a "
            "static weight matrix; merge_and_unload is not supported."
        )

    @staticmethod
    def unmerge(module, active_adapter, orig_weight):
        raise NotImplementedError(
            "DISeL gates are input-dependent; nothing to unmerge."
        )

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        gate = module.disel_gate[active_adapter]

        x_dropped = dropout(x)
        # Run the gate in the same dtype as the LoRA matrices.
        target_dtype = lora_A.weight.dtype
        a_out = lora_A(x_dropped)
        gate_values = gate(x_dropped.to(target_dtype))
        delta = lora_B(gate_values * a_out) * scaling
        return result + delta.to(result.dtype)

"""Glue between :class:`DiselConfig` and a PEFT-wrapped model.

Two public entry points:

* :func:`enable_disel` walks a ``PeftModel`` produced by
  :func:`peft.get_peft_model`, attaches a per-rank gate to every
  :class:`peft.tuners.lora.layer.LoraLayer`, and registers
  :class:`disel.variant.DiselLinearVariant` for the active adapter so that
  PEFT's forward and save/load paths route through the DISeL computation.

* :func:`build_optimizer` constructs an ``AdamW`` with three parameter groups
  matching the original paper: weight-decay matrices, no-decay biases /
  LayerNorms, and gate parameters (separate learning rate, weight decay
  disabled so the gate cannot be driven to zero).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer

from .config import DiselConfig
from .layer import LightRankGate, RankGate
from .variant import DiselLinearVariant

__all__ = ["enable_disel", "build_optimizer", "GATE_PARAM_KEY"]

# Name under which gate ModuleDicts are stored on each LoraLayer. Matches the
# entry added to `adapter_layer_names` by :class:`DiselLinearVariant.init`, so
# PEFT's `set_peft_model_state_dict` round-trips the gate parameters.
GATE_PARAM_KEY = "disel_gate"


def _make_gate(in_features: int, rank: int, config: DiselConfig) -> nn.Module:
    if config.disel_gate_bottleneck_dim is None:
        return RankGate(
            in_features=in_features,
            rank=rank,
            bias_init=config.disel_gate_bias_init,
            normalize=config.disel_gate_normalize,
            weight_init=config.disel_gate_weight_init,
        )
    return LightRankGate(
        in_features=in_features,
        rank=rank,
        bottleneck_dim=config.disel_gate_bottleneck_dim,
        bias_init=config.disel_gate_bias_init,
        normalize=config.disel_gate_normalize,
        weight_init=config.disel_gate_weight_init,
    )


def enable_disel(model: PeftModel, config: DiselConfig, adapter_name: str = "default") -> PeftModel:
    """Attach DISeL gates to every LoRA layer in ``model``.

    Call this immediately after :func:`peft.get_peft_model`. It is idempotent:
    re-running it on a model that already has DISeL attached is a no-op for
    that adapter.

    Args:
        model: The PEFT-wrapped model (output of ``get_peft_model``).
        config: The :class:`DiselConfig` used to build the model. Gate hyper-
            parameters are read from this object.
        adapter_name: Adapter for which gates should be attached. Multi-adapter
            scenarios should call this once per adapter.
    """
    if not isinstance(config, DiselConfig):
        raise TypeError(
            f"enable_disel expected DiselConfig, got {type(config).__name__}"
        )

    base_dtype = next(model.parameters()).dtype
    base_device = next(model.parameters()).device

    n_attached = 0
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        if adapter_name not in module.lora_A:
            continue
        if not hasattr(module, GATE_PARAM_KEY):
            module.add_module(GATE_PARAM_KEY, nn.ModuleDict())
        gate_dict: nn.ModuleDict = getattr(module, GATE_PARAM_KEY)
        if adapter_name in gate_dict:
            continue  # already enabled
        rank = module.r[adapter_name]
        in_features = module.in_features
        gate = _make_gate(in_features, rank, config).to(device=base_device, dtype=base_dtype)
        gate_dict[adapter_name] = gate
        module.lora_variant[adapter_name] = DiselLinearVariant
        # Register the param-dict name on the layer so PEFT's state-dict
        # plumbing picks the gates up on save/load.
        DiselLinearVariant.init(module, adapter_name)
        n_attached += 1

    if n_attached == 0:
        raise RuntimeError(
            f"enable_disel found no LoRA layers carrying adapter {adapter_name!r}. "
            "Did you forget to call peft.get_peft_model first, or did target_modules "
            "match nothing?"
        )

    # Trainable parameters need to be marked so they receive gradients and
    # so that PEFT's `print_trainable_parameters` reports them.
    for name, param in model.named_parameters():
        if f".{GATE_PARAM_KEY}." in name:
            param.requires_grad_(True)

    return model


def _iter_named_trainable(model: nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def build_optimizer(
    model: nn.Module,
    base_lr: float,
    gate_lr: float | None = None,
    gate_lr_multiplier: float | None = None,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Return an :class:`AdamW` with the three parameter groups DISeL needs.

    Groups:

    1. Decay matrices: weight decay = ``weight_decay``, LR = ``base_lr``.
    2. No-decay parameters (biases, LayerNorm / RMSNorm gains): weight decay = 0,
       LR = ``base_lr``. Standard HuggingFace ``Trainer`` convention.
    3. Gate parameters (``disel_gate.*``): weight decay = 0 (so the gate
       projection cannot be driven to zero), LR = ``gate_lr`` or
       ``base_lr * gate_lr_multiplier``.

    Exactly one of ``gate_lr`` / ``gate_lr_multiplier`` should be provided;
    if neither is, the multiplier defaults to 5 (matches the paper recipe,
    e.g. base 2e-4 / gate 1e-3).
    """
    if gate_lr is not None and gate_lr_multiplier is not None:
        raise ValueError("Pass either gate_lr or gate_lr_multiplier, not both.")
    if gate_lr is None and gate_lr_multiplier is None:
        gate_lr_multiplier = 5.0
    if gate_lr is None:
        gate_lr = base_lr * gate_lr_multiplier

    # Names that should NOT receive weight decay. We follow HuggingFace's
    # convention: biases and any 1-D normaliser gain (LayerNorm, RMSNorm).
    no_decay_suffixes = ("bias",)
    no_decay_modules = (nn.LayerNorm,)
    # Also catch RMSNorm-style modules where the parameter name ends in
    # ".weight" but the module is a normalisation layer.
    norm_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, no_decay_modules) or module.__class__.__name__.endswith("Norm"):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []

    for name, param in _iter_named_trainable(model):
        if f".{GATE_PARAM_KEY}." in name:
            gate_params.append(param)
        elif id(param) in norm_param_ids or name.endswith(no_decay_suffixes):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay, "lr": base_lr})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr})
    if gate_params:
        groups.append({"params": gate_params, "weight_decay": 0.0, "lr": gate_lr})

    if not groups:
        raise RuntimeError("build_optimizer found no trainable parameters in the model.")

    return torch.optim.AdamW(groups, betas=betas, eps=eps)

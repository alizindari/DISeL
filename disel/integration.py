"""Glue between :class:`DiselConfig` and a PEFT-wrapped model.

Public entry points:

* :func:`enable_disel` walks a ``PeftModel`` produced by
  :func:`peft.get_peft_model`, attaches a per-rank gate to every
  :class:`peft.tuners.lora.layer.LoraLayer`, and registers
  :class:`disel.variant.DiselLinearVariant` for the active adapter.
* :func:`build_optimizer` constructs the three-group ``AdamW`` used in the
  paper (weight-decay matrices, no-decay biases/LayerNorms, gate group with
  separate LR and no weight decay).
* :func:`load_gate_state_dict` loads the gate tensors from a saved adapter
  directory into a model that already has DISeL attached.
* :func:`from_pretrained` is a one-call loader that wraps PEFT's
  ``PeftModel.from_pretrained`` + ``enable_disel`` + ``load_gate_state_dict``.

A short note on persistence. Vanilla PEFT does not know about DISeL, so the
naive sequence ``PeftModel.from_pretrained(base, path)`` builds LoRA layers
*without* the ``lora_disel_gate`` ModuleDicts, and PEFT's state-dict loader
silently drops the gate keys it doesn't have a destination for. We therefore
do the load in three explicit steps: build PEFT layers, attach gates with
``enable_disel``, then re-apply the saved state dict to fill the gate
parameters. :func:`from_pretrained` does this for you.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer

from .config import DiselConfig
from .layer import LightRankGate, RankGate
from .variant import DiselLinearVariant

__all__ = [
    "GATE_PARAM_KEY",
    "build_optimizer",
    "enable_disel",
    "from_pretrained",
    "load_gate_state_dict",
]

# Name under which gate ModuleDicts are stored on each LoraLayer. Matches the
# entry added to `adapter_layer_names` by :class:`DiselLinearVariant.init`, so
# PEFT's `set_peft_model_state_dict` round-trips the gate parameters.
GATE_PARAM_KEY = "lora_disel_gate"


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


def _read_adapter_state(adapter_path: Path) -> dict[str, torch.Tensor]:
    sft = adapter_path / "adapter_model.safetensors"
    bin_ = adapter_path / "adapter_model.bin"
    if sft.is_file():
        from safetensors.torch import load_file
        return load_file(str(sft))
    if bin_.is_file():
        return torch.load(bin_, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"no adapter_model.safetensors or adapter_model.bin in {adapter_path}"
    )


def load_gate_state_dict(
    model: PeftModel,
    adapter_path: Union[str, Path],
    adapter_name: str = "default",
) -> PeftModel:
    """Load gate weights from a saved adapter directory into ``model``.

    The model must already have DISeL attached (i.e. :func:`enable_disel` must
    have run, so the ``lora_disel_gate`` ModuleDicts exist on every LoRA
    layer). This function reads ``adapter_model.safetensors`` (or
    ``adapter_model.bin``) from ``adapter_path``, filters for keys containing
    ``GATE_PARAM_KEY``, and applies them with ``load_state_dict(strict=False)``.

    Raises ``ValueError`` if the checkpoint has no gate keys (the saved model
    was not trained with DISeL), and ``RuntimeError`` if there are gate keys
    in the checkpoint that have no destination in the model (likely an
    adapter-name mismatch).
    """
    adapter_path = Path(adapter_path)
    state = _read_adapter_state(adapter_path)
    gate_state = {k: v for k, v in state.items() if GATE_PARAM_KEY in k}
    if not gate_state:
        raise ValueError(
            f"No '{GATE_PARAM_KEY}' keys in {adapter_path}; the saved adapter "
            "does not appear to have been trained with DISeL."
        )

    missing, unexpected = model.load_state_dict(gate_state, strict=False)
    unexpected_gate_keys = [k for k in unexpected if GATE_PARAM_KEY in k]
    if unexpected_gate_keys:
        raise RuntimeError(
            f"{len(unexpected_gate_keys)} gate key(s) in the checkpoint have "
            "no destination in the model — did you forget to call enable_disel "
            "first, or pass the wrong adapter_name? Examples: "
            f"{unexpected_gate_keys[:3]}"
        )

    # Sanity-check: every layer's gate should now have a loaded weight.
    loaded_layers = {k.rsplit(f".{GATE_PARAM_KEY}.", 1)[0] for k in gate_state}
    model_layers = {
        n.rsplit(f".{GATE_PARAM_KEY}.", 1)[0]
        for n, _ in model.named_parameters()
        if GATE_PARAM_KEY in n
    }
    diff = model_layers - loaded_layers
    if diff:
        warnings.warn(
            f"{len(diff)} layer(s) have DISeL gates with no matching entry in "
            f"the checkpoint (they keep their fresh init). Examples: "
            f"{sorted(diff)[:3]}"
        )
    return model


def from_pretrained(
    base_model: nn.Module,
    adapter_path: Union[str, Path],
    config: Optional[DiselConfig] = None,
    adapter_name: str = "default",
    **kwargs,
) -> PeftModel:
    """Load a DISeL-gated PEFT model from a saved adapter directory.

    Equivalent to::

        peft_model = PeftModel.from_pretrained(base_model, adapter_path, ...)
        enable_disel(peft_model, config)
        load_gate_state_dict(peft_model, adapter_path)

    in a single call, with the saved ``DiselConfig`` auto-loaded from
    ``adapter_config.json`` when ``config`` is ``None``. Extra ``**kwargs`` are
    forwarded to :meth:`peft.PeftModel.from_pretrained` (e.g. ``device_map``,
    ``torch_dtype``, ``is_trainable``).

    Args:
        base_model: The base ``nn.Module`` (e.g. output of
            ``AutoModelForCausalLM.from_pretrained``).
        adapter_path: Directory written by ``model.save_pretrained(...)`` at
            training time.
        config: The :class:`DiselConfig` used at training time. If ``None``,
            we read it from ``adapter_path/adapter_config.json``. Provide one
            explicitly only if you want to override gate hyperparameters at
            load time (rare).
        adapter_name: PEFT adapter name.
    """
    adapter_path = Path(adapter_path)
    if config is None:
        # PEFT's PeftConfig.from_pretrained dispatches on the `peft_type` field
        # in adapter_config.json, which is "LORA", so it would build a plain
        # LoraConfig and drop our extra fields. Read the JSON ourselves.
        import inspect
        import json
        cfg_path = adapter_path / "adapter_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"{cfg_path} not found; cannot infer DiselConfig automatically. "
                "Pass `config=DiselConfig(...)` explicitly."
            )
        raw = json.loads(cfg_path.read_text())
        # DiselConfig inherits LoraConfig's fields; drop any keys it doesn't
        # accept (e.g. PEFT runtime metadata) and forward the rest.
        valid = set(inspect.signature(DiselConfig).parameters)
        config = DiselConfig(**{k: v for k, v in raw.items() if k in valid})
    elif not isinstance(config, DiselConfig):
        raise TypeError(
            f"from_pretrained expected DiselConfig or None, "
            f"got {type(config).__name__}"
        )

    peft_model = PeftModel.from_pretrained(
        base_model, str(adapter_path), adapter_name=adapter_name, **kwargs,
    )
    enable_disel(peft_model, config, adapter_name=adapter_name)
    load_gate_state_dict(peft_model, adapter_path, adapter_name=adapter_name)
    return peft_model

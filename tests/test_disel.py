"""Sanity tests for the DISeL package.

These run on CPU against a tiny HuggingFace model. They cover the same matrix
PEFT's ``tests/test_custom_models.py`` enforces for in-tree adapters:

* gates start nearly closed (output close to base model at init);
* gradient reaches the gate parameters;
* save_pretrained / from_pretrained round-trips the gates;
* disable_adapter recovers the base output;
* merge_and_unload raises NotImplementedError.

Run with ``pytest -q``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from peft import PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

import disel

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _build_model(use_gate: bool = True):
    base = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    base.eval()
    config = disel.DiselConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        disel_gate_bias_init=-3.0,
    )
    peft_model = get_peft_model(base, config)
    if use_gate:
        disel.enable_disel(peft_model, config)
    return peft_model, config


def _dummy_batch(seq_len: int = 8, vocab_size: int | None = None) -> torch.Tensor:
    if vocab_size is None:
        vocab_size = AutoConfig.from_pretrained(TINY_MODEL).vocab_size
    return torch.randint(0, vocab_size, (2, seq_len))


def test_disel_starts_nearly_at_base():
    """At init the gate values are nearly zero, so a DISeL forward should be
    very close to the base model's output (and at most as far from base as a
    vanilla LoRA forward at the same rank, which already starts close because
    lora_B is zero-initialised)."""
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(TINY_MODEL).eval()
    inputs = _dummy_batch()

    with torch.no_grad():
        base_logits = base(inputs).logits

    disel_model, _ = _build_model(use_gate=True)
    disel_model.eval()
    with torch.no_grad():
        disel_logits = disel_model(inputs).logits

    plain, _ = _build_model(use_gate=False)
    plain.eval()
    with torch.no_grad():
        plain_logits = plain(inputs).logits

    base_to_disel = (base_logits - disel_logits).abs().mean().item()
    base_to_plain = (base_logits - plain_logits).abs().mean().item()
    # Both should be small because lora_B is zero-initialised, but the DISeL
    # residual is further attenuated by sigmoid(-3) ≈ 0.047 — assert that the
    # DISeL output is at most as far from base as the plain LoRA output.
    assert base_to_disel <= base_to_plain + 1e-6


def test_gradient_reaches_gate_parameters():
    """Gate parameters receive gradient *after* lora_B becomes non-zero.

    With PEFT's default lora_B = 0 init, ∂L/∂W_g ∝ lora_B^T · (...) is exactly
    zero at step 0. We therefore take one optimizer step to lift lora_B off
    zero, then verify that the next backward propagates a finite, non-zero
    gradient into every gate parameter.
    """
    model, _ = _build_model()
    model.train()
    inputs = _dummy_batch()
    opt = disel.build_optimizer(model, base_lr=1e-2, gate_lr=1e-2)

    # Step 1: pushes lora_B off zero via its A-side gradient path.
    out = model(inputs, labels=inputs)
    out.loss.backward()
    opt.step()
    opt.zero_grad()

    # Step 2: now the gate is on the computation path with non-zero lora_B.
    out = model(inputs, labels=inputs)
    out.loss.backward()

    gate_grads = [
        (name, p.grad)
        for name, p in model.named_parameters()
        if f".{disel.GATE_PARAM_KEY}." in name
    ]
    assert gate_grads, "no gate parameters were trained"
    for name, grad in gate_grads:
        assert grad is not None, f"no grad on {name}"
        assert torch.isfinite(grad).all(), f"non-finite grad on {name}"
        assert grad.abs().sum() > 0, f"zero grad on {name}"


def test_save_load_round_trip(tmp_path: Path):
    """Round-trip a trained model and verify the gate weights actually persist.

    Naively comparing logits at init is not enough: with lora_B = 0 the model
    output is independent of the gate values, so a buggy save/load that drops
    the gates would still pass. We therefore take an optimizer step (which
    moves both lora_B and the gates off their init), save, rebuild, and
    require the saved gate parameters to load back bit-exactly.
    """
    torch.manual_seed(0)
    model, config = _build_model()
    model.train()
    inputs = _dummy_batch()

    # Two steps so the gates move off their init: step 1 nudges lora_B off
    # zero (which gives the gates a non-zero gradient path), step 2 actually
    # moves W_g / b_g.
    opt = disel.build_optimizer(model, base_lr=1e-2, gate_lr=1e-2)
    for _ in range(2):
        out = model(inputs, labels=inputs)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

    # Snapshot the gate parameters and a reference forward pass.
    gate_state_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if f".{disel.GATE_PARAM_KEY}." in name
    }
    assert gate_state_before, "no gate parameters present in trained model"
    # Confirm the gates have actually moved (otherwise the test is vacuous).
    for name, p in gate_state_before.items():
        if name.endswith(".bias"):
            assert not torch.allclose(p, torch.full_like(p, -3.0)), (
                f"{name} did not move from its init"
            )

    model.eval()
    with torch.no_grad():
        before = model(inputs).logits.clone()

    model.save_pretrained(tmp_path)

    # Verify the gates actually made it into the safetensors file before we
    # try to load — this would have caught the original `disel_gate` bug.
    from safetensors.torch import load_file
    state = load_file(tmp_path / "adapter_model.safetensors")
    saved_gate_keys = [k for k in state if disel.GATE_PARAM_KEY in k]
    assert saved_gate_keys, (
        "no gate parameters were written to adapter_model.safetensors — "
        "PEFT's state-dict filter is not picking them up"
    )

    # Rebuild from scratch using the public one-call loader.
    base = AutoModelForCausalLM.from_pretrained(TINY_MODEL).eval()
    reloaded = disel.from_pretrained(base, tmp_path)

    gate_state_after = {
        name: param.detach().clone()
        for name, param in reloaded.named_parameters()
        if f".{disel.GATE_PARAM_KEY}." in name
    }
    for name, before_p in gate_state_before.items():
        after_p = gate_state_after[name]
        assert torch.equal(before_p, after_p), (
            f"{name} did not round-trip exactly"
        )

    reloaded.eval()
    with torch.no_grad():
        after = reloaded(inputs).logits

    assert torch.allclose(before, after, atol=1e-5), (
        f"max diff after save/load = {(before - after).abs().max().item():.3e}"
    )


def test_disable_adapter_recovers_base():
    model, _ = _build_model()
    model.eval()
    inputs = _dummy_batch()
    base = AutoModelForCausalLM.from_pretrained(TINY_MODEL).eval()
    with torch.no_grad():
        base_logits = base(inputs).logits
        with model.disable_adapter():
            disabled_logits = model(inputs).logits
    assert torch.allclose(base_logits, disabled_logits, atol=1e-5)


def test_end_to_end_train_save_load_infer(tmp_path: Path):
    """End-to-end smoke test mirroring the user-facing workflow.

    Train (a few steps) → save_pretrained → fresh process: build base, then
    disel.from_pretrained → inference. Verify the inference output is the
    same as the trained model's, and that the gate parameters are NOT at
    their fresh init (i.e. the training and the reload actually transferred).
    """
    torch.manual_seed(0)
    model, _ = _build_model()
    model.train()
    inputs = _dummy_batch()

    opt = disel.build_optimizer(model, base_lr=1e-2, gate_lr=1e-2)
    for _ in range(3):
        out = model(inputs, labels=inputs)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        ref_logits = model(inputs).logits.clone()

    # Save just like a real user would.
    out_dir = tmp_path / "checkpoint"
    model.save_pretrained(out_dir)

    # Reload from scratch — the only thing carried across is `out_dir`.
    base = AutoModelForCausalLM.from_pretrained(TINY_MODEL).eval()
    reloaded = disel.from_pretrained(base, out_dir)
    reloaded.eval()

    # Sanity: the reloaded gate biases should not all be exactly -3.
    for name, p in reloaded.named_parameters():
        if name.endswith(f".{disel.GATE_PARAM_KEY}.default.linear.bias"):
            assert not torch.allclose(p, torch.full_like(p, -3.0)), (
                "gate bias is still at fresh init — load_gate_state_dict did "
                "not fire"
            )
            break
    else:
        raise AssertionError("no gate bias parameter found in reloaded model")

    with torch.no_grad():
        loaded_logits = reloaded(inputs).logits

    assert torch.allclose(ref_logits, loaded_logits, atol=1e-5), (
        f"inference differs after save/load: max diff = "
        f"{(ref_logits - loaded_logits).abs().max().item():.3e}"
    )


def test_standalone_gate_file_round_trip(tmp_path: Path):
    """Verify the optional separate-file save/load path (gate_weights.safetensors)
    used by people who prefer the localised-neurons convention of keeping
    gates separate from adapter_model.safetensors."""
    torch.manual_seed(0)
    model, _ = _build_model()
    model.train()
    inputs = _dummy_batch()

    opt = disel.build_optimizer(model, base_lr=1e-2, gate_lr=1e-2)
    for _ in range(2):
        out = model(inputs, labels=inputs)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

    snapshot = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if disel.GATE_PARAM_KEY in n
    }

    out_dir = tmp_path / "gates_only"
    written = disel.save_gate_state_dict(model, out_dir)
    assert written.is_file()
    assert written.name == disel.GATE_FILENAME

    # Wipe gates on a fresh model and reload via the standalone path.
    fresh, _ = _build_model()
    fresh.eval()
    disel.load_gate_state_dict(fresh, out_dir)

    for n, before in snapshot.items():
        after = dict(fresh.named_parameters())[n]
        assert torch.equal(before, after), f"{n} did not round-trip via standalone gate file"


def test_merge_raises_not_implemented():
    model, _ = _build_model()
    with pytest.raises(NotImplementedError):
        model.merge_and_unload()


def test_build_optimizer_groups():
    model, _ = _build_model()
    opt = disel.build_optimizer(model, base_lr=2e-4, gate_lr=1e-3)
    # Exactly one group should carry the gate params at the gate LR.
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert 1e-3 in lrs
    assert 2e-4 in lrs
    gate_group = [g for g in opt.param_groups if g["lr"] == 1e-3]
    assert len(gate_group) == 1
    assert gate_group[0]["weight_decay"] == 0.0
    assert all(p.requires_grad for p in gate_group[0]["params"])

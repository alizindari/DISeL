"""Minimal DISeL training recipe — LLaMA-2-7B on MetaMathQA.

Run on a single node with 4×A100 / H100:

    accelerate launch examples/train_metamath.py \\
        --output_dir runs/disel_llama_r64 \\
        --lora_rank 64 --learning_rate 2e-4 \\
        --gate_lr 1e-3 --gate_bias_init -3.0 \\
        --num_train_epochs 3

The script intentionally uses HuggingFace's vanilla Trainer with a custom
optimizer built by :func:`disel.build_optimizer`. The DISeL-specific code is
the four highlighted blocks (config, peft wrap, enable_disel, optimizer); the
rest is standard HF instruction-tuning boilerplate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import disel
from peft import get_peft_model

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--data_path", default="meta-math/MetaMathQA",
                    help="HF dataset id or local JSON file.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--gate_lr", type=float, default=1e-3)
    ap.add_argument("--gate_bias_init", type=float, default=-3.0)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def format_example(ex: dict, tokenizer, max_length: int) -> dict:
    instruction = ex.get("query") or ex.get("instruction") or ex["question"]
    response = ex.get("response") or ex.get("output") or ex["answer"]
    prompt = ALPACA_TEMPLATE.format(instruction=instruction)
    full = prompt + response + tokenizer.eos_token
    enc = tokenizer(full, max_length=max_length, truncation=True, padding=False)
    # Mask the instruction in the labels so loss is on the response only.
    prompt_len = len(tokenizer(prompt, max_length=max_length, truncation=True).input_ids)
    enc["labels"] = enc["input_ids"][:]
    enc["labels"][:prompt_len] = [-100] * prompt_len
    return enc


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.data_path.endswith(".json"):
        ds = load_dataset("json", data_files=args.data_path)["train"]
    else:
        ds = load_dataset(args.data_path)["train"]
    ds = ds.shuffle(seed=args.seed).train_test_split(test_size=0.05, seed=args.seed)
    train_ds = ds["train"].map(
        lambda ex: format_example(ex, tokenizer, args.max_length),
        remove_columns=ds["train"].column_names,
        desc="Tokenising train",
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    # --- DISeL: configure ---
    config = disel.DiselConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        disel_gate_bias_init=args.gate_bias_init,
    )

    # --- DISeL: wrap + attach gates ---
    model = get_peft_model(model, config)
    disel.enable_disel(model, config)
    model.print_trainable_parameters()

    # --- DISeL: optimizer with separate gate LR (no weight decay on gates) ---
    optimizer = disel.build_optimizer(
        model,
        base_lr=args.learning_rate,
        gate_lr=args.gate_lr,
        weight_decay=args.weight_decay,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,  # used by the scheduler shape only
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", label_pad_token_id=-100, padding=True
        ),
        optimizers=(optimizer, None),  # let Trainer build the cosine scheduler
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()

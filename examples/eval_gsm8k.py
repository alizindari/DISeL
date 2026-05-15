"""Minimal GSM8K evaluation for a DISeL-fine-tuned LLaMA-2-7B.

End-to-end demo of the load + inference path: takes a base model and a
directory written by `model.save_pretrained(...)` during training, reloads
via :func:`disel.from_pretrained`, runs greedy generation on GSM8K test
problems, and reports exact-match accuracy on the final numerical answer.

Usage:
    python examples/eval_gsm8k.py \\
        --base meta-llama/Llama-2-7b-hf \\
        --adapter runs/disel_llama_r64 \\
        --num_problems 200

For the full paper-style evaluation (target task + 14-benchmark retention
suite), use `lm-evaluation-harness`; this script is intended as a quick
smoke check to confirm the saved adapter actually does what it should.
"""

from __future__ import annotations

import argparse
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import disel

ALPACA_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """Pull the final numerical answer out of a GSM8K-style response."""
    m = GSM8K_ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1]
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-2-7b-hf",
                    help="Base model HuggingFace id or local path.")
    ap.add_argument("--adapter", required=True,
                    help="Directory containing adapter_model.safetensors "
                         "(written by model.save_pretrained at training time).")
    ap.add_argument("--num_problems", type=int, default=100,
                    help="How many GSM8K test problems to evaluate. "
                         "Pass 0 for the full 1319-problem test set.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
    )
    print(f"Loading DISeL adapter: {args.adapter}")
    model = disel.from_pretrained(base, args.adapter)
    model.eval().to(args.device)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_problems > 0:
        ds = ds.select(range(min(args.num_problems, len(ds))))
    n = len(ds)
    print(f"Evaluating on {n} GSM8K test problems...")

    correct = 0
    for i, ex in enumerate(ds):
        prompt = ALPACA_TEMPLATE.format(instruction=ex["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        pred = extract_final_number(response)
        gold = extract_final_number(ex["answer"])
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}] running accuracy = {100 * correct / (i+1):.2f}%")

    acc = 100 * correct / n
    print(f"\nGSM8K exact-match accuracy: {correct}/{n} = {acc:.2f}%")


if __name__ == "__main__":
    main()

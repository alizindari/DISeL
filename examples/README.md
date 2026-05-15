# Examples

Paper-reproducible recipes live here. Each script uses the public DISeL API
(`DiselConfig`, `enable_disel`, `build_optimizer`) plus vanilla HuggingFace
Trainer; the four DISeL-specific lines are highlighted in the source.

## `train_metamath.py`

LLaMA-2-7B on MetaMathQA. Hyperparameters match the paper:

| Field | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Base LR | 2e-4 (cosine, 2% warmup) |
| Gate LR | 1e-3, weight decay disabled |
| Gate bias init | −3.0 (σ(−3) ≈ 0.047) |
| Effective batch size | 32 |
| Max sequence length | 512 |
| Epochs | 3 |

Launch (single node, 4×A100):

```bash
accelerate launch examples/train_metamath.py \
    --output_dir runs/disel_llama_r64 \
    --lora_rank 64 --lora_alpha 128 \
    --learning_rate 2e-4 --gate_lr 1e-3 --gate_bias_init -3.0 \
    --num_train_epochs 3
```

## `eval_gsm8k.py`

Smoke-test the load + inference path on a checkpoint produced by
`train_metamath.py`. Reloads the model via `disel.from_pretrained`, runs
greedy decoding on `N` GSM8K test problems, and prints exact-match accuracy.

```bash
python examples/eval_gsm8k.py \
    --base meta-llama/Llama-2-7b-hf \
    --adapter runs/disel_llama_r64 \
    --num_problems 200
```

For the full 14-benchmark retention suite used in the paper, use
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).
The harness's built-in `--model peft` path will not work directly — it calls
`PeftModel.from_pretrained` without our `enable_disel` / `load_gate_state_dict`
steps, so the gates would silently stay at their fresh init. Use the
harness's in-process API instead and pass the already-loaded DISeL model:

```python
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import disel, torch

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16,
)
model = disel.from_pretrained(base, "runs/disel_llama_r64")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

results = simple_evaluate(
    model=HFLM(pretrained=model, tokenizer=tok),
    tasks=["gsm8k", "hellaswag", "winogrande", "arc_challenge", "arc_easy",
           "lambada_openai", "boolq", "piqa", "triviaqa", "openbookqa",
           "sciq", "mmlu", "medqa_4options", "commonsense_qa", "nq_open"],
    num_fewshot=5,
    batch_size="auto",
)
```

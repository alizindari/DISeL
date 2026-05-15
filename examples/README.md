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

# 02 — SFT → RM → DPO alignment chain

Takes example 01's pretrained checkpoint and runs the **full preference-based
alignment chain** on a synthetic task: "Which Shakespeare play features
<character>?".

## Pipeline

1. **Generate SFT data** (`out/sft.jsonl`, 180 train + 20 eval). Each example:
   `{"prompt": "Q: Which play features Romeo?\nA:", "response": " Romeo and Juliet"}`
2. **SFT** for 200 steps on the BPE-tokenized base model (lr 1e-4, batch 4, seq 128). Uses `platform.alignment._common.tokenize_and_pack` for prompt/response masking.
3. **Generate preference data** (`out/prefs.jsonl`, 100 pairs): chosen = correct mapping, rejected = same prompt with a wrong play swapped in.
4. **Reward model** — builds a `RewardModel` wrapping a fresh trunk initialised from the SFT weights, trains with Bradley-Terry loss for 200 steps.
5. **DPO** — runs `dpo_loss` for 200 steps on the SFT model with a frozen reference copy.
6. **Comparison**: for 10 held-out test prompts, generate from base / SFT / DPO models via `TorchEngine` (greedy, 20 new tokens). For each model+prompt: response, length, RM score, exact-match accuracy.

Why hand-rolled and not `run_sft()` / `run_dpo()` / `train_reward_model()`?
Those convenience helpers in `platform.alignment.*` hard-code
`BytesTokenizer()` (vocab=512). We trained a vocab=4096 BPE so we use the
lower-level primitives (`tokenize_and_pack`, `compute_logps`, `dpo_loss`,
`bt_loss`, `RewardModel`) directly. Same algorithms, real tokenizer.

## Run

```bash
bash run.sh                   # requires examples/01 to have run first
```

Outputs:
- `out/sft.jsonl`, `out/eval_sft.jsonl`, `out/prefs.jsonl`, `out/test.jsonl`
- `out/sft_model.pt`, `out/rm.pt`, `out/dpo_model.pt`
- `result.md` — comparison table + summary stats

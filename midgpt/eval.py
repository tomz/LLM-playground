"""Eval harness: validation perplexity + HellaSwag zero-shot accuracy."""
import argparse, os
import torch
import tiktoken
from model import GPT, GPTConfig


@torch.no_grad()
def hellaswag_acc(model, enc, device, max_examples: int = 1000) -> float:
    """HF datasets `hellaswag` validation. Score = avg log-prob of completion tokens."""
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    correct = total = 0
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ctx_ids = enc.encode_ordinary(ex["ctx"])
        scores = []
        for ending in ex["endings"]:
            end_ids = enc.encode_ordinary(" " + ending)
            ids = torch.tensor([ctx_ids + end_ids], device=device)
            ids = ids[:, : model.cfg.block_size]
            logits, _ = model(ids)
            # full logits at every position: re-run with targets to get per-token loss
            inp = ids[:, :-1]; tgt = ids[:, 1:]
            logits, _ = model(inp, tgt)
            logp = torch.log_softmax(logits, dim=-1)
            tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            # score only completion tokens
            n_end = min(len(end_ids), tok_logp.size(1))
            scores.append(tok_logp[0, -n_end:].mean().item())
        pred = max(range(4), key=lambda i: scores[i])
        correct += int(pred == int(ex["label"]))
        total += 1
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="ppl")  # comma-sep: ppl,hellaswag
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-examples", type=int, default=1000)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    sd = torch.load(args.ckpt, map_location=device)
    cfg = sd["cfg"]
    model = GPT(GPTConfig(**cfg["model"])).to(device).eval()
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)
    enc = tiktoken.get_encoding(cfg["tokenizer"])

    tasks = set(args.tasks.split(","))
    if "ppl" in tasks:
        from data import ShardDataset
        ds = ShardDataset(os.path.join("data", cfg["dataset"]), cfg["model"]["block_size"], device)
        gen = torch.Generator(); gen.manual_seed(0)
        losses = []
        for _ in range(200):
            x, y = ds.get_batch(cfg["train"]["micro_batch"], gen)
            _, loss = model(x, y); losses.append(loss.item())
        L = sum(losses) / len(losses)
        print(f"val_loss = {L:.4f}   ppl = {torch.tensor(L).exp().item():.3f}")
    if "hellaswag" in tasks:
        acc = hellaswag_acc(model, enc, device, args.max_examples)
        print(f"hellaswag = {acc*100:.2f}%  (n={args.max_examples})")


if __name__ == "__main__":
    main()

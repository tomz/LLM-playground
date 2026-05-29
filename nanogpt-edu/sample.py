"""Generate text from a checkpoint."""
import argparse, torch
from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg, meta = sd["cfg"], sd["meta"]
    mcfg = GPTConfig(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_kv_head=cfg["n_kv_head"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"],
        dropout=0.0, rope_base=cfg["rope_base"],
        # carry the speedrun knobs so the architecture matches the checkpoint
        qk_norm=cfg.get("qk_norm", False),
        zero_init_proj=cfg.get("zero_init_proj", False),
        tie_embeddings=cfg.get("tie_embeddings", True),
    )
    model = GPT(mcfg).to(device).eval()
    # tolerate torch.compile-wrapped state dicts
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)

    # Two tokenizer regimes share the same shard format:
    #   * char-level  → meta has stoi/itos dicts (prepare_shakespeare.py)
    #   * BPE (GPT-2) → meta["tokenizer"] == "gpt2" (prepare_fineweb.py)
    if meta.get("tokenizer") == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        ids = torch.tensor([enc.encode(args.prompt)], dtype=torch.long, device=device)
        out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()
        print(enc.decode(out))
    else:
        stoi, itos = meta["stoi"], meta["itos"]
        ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=device)
        out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()
        print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()

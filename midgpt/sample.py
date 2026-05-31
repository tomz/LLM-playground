"""Generate text from a midgpt checkpoint.

Sampling modes:
  * ``--temperature 0``  → greedy (argmax). Deterministic.
  * ``--top-k K``        → multinomial over the top-K logits.
  * ``--top-p P``        → nucleus over the smallest prefix summing to P.
  * Both ``--top-k`` and ``--top-p`` may be combined; ``top-k`` is applied first.
"""
import argparse, torch, tiktoken
from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="0 = greedy; >0 = multinomial.")
    ap.add_argument("--top-k", type=int, default=200,
                    help="0 or negative disables top-k.")
    ap.add_argument("--top-p", type=float, default=None,
                    help="Nucleus cutoff in (0, 1). Default: disabled.")
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, seed RNG for reproducible non-greedy samples.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = sd["cfg"]
    model = GPT(GPTConfig(**cfg["model"])).to(device).eval()
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)
    enc = tiktoken.get_encoding(cfg["tokenizer"])

    ids = torch.tensor([enc.encode_ordinary(args.prompt)], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    out = model.generate(
        ids, args.max_new_tokens,
        temperature=args.temperature, top_k=top_k, top_p=args.top_p,
    )[0].tolist()
    print(enc.decode(out))


if __name__ == "__main__":
    main()

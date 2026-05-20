"""Generate text from a midgpt checkpoint."""
import argparse, torch, tiktoken
from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = sd["cfg"]
    model = GPT(GPTConfig(**cfg["model"])).to(device).eval()
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)
    enc = tiktoken.get_encoding(cfg["tokenizer"])

    ids = torch.tensor([enc.encode_ordinary(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()
    print(enc.decode(out))


if __name__ == "__main__":
    main()

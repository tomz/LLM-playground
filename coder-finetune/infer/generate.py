"""Generate code from a fine-tuned model."""
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    msgs = [
        {"role": "system", "content": "You are a helpful Python coding assistant."},
        {"role": "user", "content": args.prompt},
    ]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = args.prompt
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0),
            temperature=max(args.temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()

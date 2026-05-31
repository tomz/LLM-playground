"""Generate code from a fine-tuned model.

Handles three cases automatically:
  1. Full-FT model — `args.model` is a HF model dir (config.json + weights).
  2. LoRA adapter — `args.model` is an adapter dir (adapter_config.json) and we
     resolve the base model from the adapter config and load via PEFT.
  3. Already-merged model — same as (1).

Prompt style
------------
`--prompt-style chat` (default) wraps the prompt in the tokenizer's chat
template. This is the right choice for *Instruct* base models. For raw
*Base* models that haven't seen ChatML (e.g. `Qwen/Qwen2.5-Coder-0.5B`),
pass `--prompt-style raw` to feed the prompt directly — otherwise the
model just hallucinates.
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _eos_ids(tok) -> list[int]:
    """Collect all stop-token ids the chat template might emit. We need to
    cover the model's EOS *and* the chat-template turn-end (<|im_end|> for
    Qwen / ChatML), otherwise generation often runs past the answer."""
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        try:
            tid = tok.convert_tokens_to_ids(marker)
            # convert_tokens_to_ids returns None for unknown tokens, or the
            # tokenizer's `unk_token_id` — both are useless as stop ids.
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                ids.add(tid)
        except Exception:
            pass
    return sorted(ids)


def _load(model_path: str, device: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = Path(model_path)
    is_adapter = (p / "adapter_config.json").exists()
    # transformers' load kwarg name changed: pre-4.46 only accepts ``torch_dtype``;
    # 4.46+/5.x accept both (``dtype`` is the new spelling, ``torch_dtype`` still
    # works as an alias). Our requirements.txt allows `>=4.45,<5.0`, so the older
    # spelling is the safe one for the full pin range — merge_lora.py + train.py
    # already use it. (The previous ``dtype=`` form would TypeError on 4.45.)
    if is_adapter:
        cfg = json.loads((p / "adapter_config.json").read_text())
        base_name = cfg.get("base_model_name_or_path")
        if not base_name:
            raise ValueError(f"adapter at {p} has no base_model_name_or_path")
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompt-style", choices=["chat", "raw"], default="chat",
                    help="'chat' wraps in tokenizer chat template (Instruct models); "
                         "'raw' feeds the prompt verbatim (Base models)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok, model = _load(args.model, device, dtype)

    if args.prompt_style == "chat":
        msgs = [
            {"role": "system", "content": "You are a helpful Python coding assistant."},
            {"role": "user", "content": args.prompt},
        ]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = args.prompt
    else:
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
            eos_token_id=_eos_ids(tok) or tok.eos_token_id,
        )
    print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()

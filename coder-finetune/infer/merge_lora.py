"""Merge a LoRA adapter into the base model and save in HF format.
Needed before serving with vLLM (which can also load LoRA directly).
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model name or path")
    ap.add_argument("--adapter", required=True, help="path to LoRA adapter dir")
    ap.add_argument("--out", required=True, help="output dir for merged weights")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.adapter)
    tok.save_pretrained(args.out)
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()

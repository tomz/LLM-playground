"""`python -m distgpt.cli train|eval|sample`"""
import argparse, sys
import yaml


def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_train(args):
    from .training.trainer import train
    train(_load(args.config), data_dir_override=args.data)


def cmd_eval(args):
    if args.lm_eval_tasks:
        # HF export + lm-evaluation-harness path. Heavier deps, so lazily
        # imported via the runner module.
        from .eval.lm_eval_runner import run_lm_eval
        tasks = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]
        run_lm_eval(
            config_path=args.config, ckpt_path=args.ckpt, tasks=tasks,
            tokenizer_dir=args.tokenizer_dir,
            num_fewshot=args.num_fewshot, limit=args.limit,
            batch_size=args.batch_size,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            output_path=args.output_path,
        )
        return
    # Default path: in-cluster held-out loss / perplexity over args.n batches.
    import torch
    from .model.config import ModelConfig
    from .model.transformer import GPT
    from .data.streaming import StreamingLoader
    from .eval.harness import held_out_loss
    cfg = _load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd if not isinstance(sd, dict) or "model" not in sd else sd["model"])
    loader = StreamingLoader(args.data, cfg["data"]["seq_len"], cfg["train"]["micro_batch"],
                             rank=0, world_size=1, seed=0, device=device)
    print(held_out_loss(model, loader, n_batches=args.n))


def cmd_sample(args):
    print("sample CLI not implemented; use distgpt as a library after training.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="distgpt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None, help="override data.dir")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=None,
                    help="data dir (required for held-out loss; ignored for lm-eval)")
    p.add_argument("--n", type=int, default=50,
                    help="number of batches for held-out loss")
    # lm-eval-harness path (opt-in via --lm-eval-tasks)
    p.add_argument("--lm-eval-tasks", default=None,
                    help="comma-separated lm-eval task names (e.g. hellaswag,arc_easy). "
                         "When set, switches to HF export + lm-eval-harness path.")
    p.add_argument("--tokenizer-dir", default=None,
                    help="dir with tokenizer.json etc. for HF export; "
                         "falls back to GPT-2 BPE if missing.")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                    help="limit examples per task (for quick smoke tests)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output-path", default=None,
                    help="write lm-eval results JSON here")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("sample")
    p.set_defaults(fn=cmd_sample)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

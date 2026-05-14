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
    import torch
    from .model.config import ModelConfig
    from .model.transformer import GPT
    from .data.streaming import StreamingLoader
    from .eval.harness import held_out_loss
    cfg = _load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    sd = torch.load(args.ckpt, map_location=device)
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
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=50)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("sample")
    p.set_defaults(fn=cmd_sample)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

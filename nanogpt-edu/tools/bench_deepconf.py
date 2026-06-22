"""Eval/serving benchmark: DeepConf — test-time confidence filtering of traces.

DeepConf (Fu, Wang, Tian, Zhao — Meta AI / UCSD, Aug 2025; arXiv 2508.15260) is a
*parallel-thinking* decoder that uses the model's **own** token-confidence — the
per-token logprob of the sampled token, aggregated over a sliding window — to
drop low-confidence reasoning traces. It buys *more accuracy for fewer tokens*
with **no training and no extra hyperparameters**, and slots into an existing
sampling loop. Two rungs:

  * **offline**  — sample k traces, then take a **confidence-weighted vote**
    instead of a plain majority vote (the trivially-correct first rung).
  * **online**   — **early-abort** a trace the moment its sliding-window
    confidence drops below a floor calibrated from a small warmup set (the
    token-saving rung).

This tool measures both against the plain self-consistency (majority-vote)
baseline on a toy deterministic task, reporting accuracy and the token savings:

    python tools/bench_deepconf.py --ckpt out/tiny/ckpt.pt --traces 16
    python tools/bench_deepconf.py --task parity --traces 32 --online

The *mechanism* and the token-accounting are the point — exactly as in the MTP
speculative bench (`tools/bench_mtp_spec.py`), this is the test-time mirror of
the train-time entropy work (frontier-platform's `EntropyController`): same
signal (token confidence / entropy), opposite end of the pipeline. With a real
reasoning checkpoint the offline vote also lifts accuracy; with the random
fallback model the harness still demonstrates the token-savings bookkeeping.

It uses only the model's existing `forward()` — no architecture change. The pure
helpers (`token_confidences`, `lowest_group_confidence`, `confidence_weighted_vote`)
are import-friendly so the unit test can pin the math on CPU without a model.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure confidence helpers (no model needed — unit-tested directly).
# --------------------------------------------------------------------------- #
def token_confidences(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Per-token confidence = log P(sampled token) under the step's distribution.

    DeepConf's atomic signal. ``logits`` is [T, V] (the next-token distribution
    at each of T generated steps); ``token_ids`` is [T] (the token actually
    emitted at each step). Returns [T] of logprobs (higher = more confident).
    A near-equivalent variant uses negative entropy of the full distribution;
    we use the sampled-token logprob because it's what online early-abort can
    read with zero extra compute.
    """
    logprobs = F.log_softmax(logits.float(), dim=-1)            # [T, V]
    return logprobs.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1)  # [T]


def sliding_group_confidence(conf: torch.Tensor, window: int) -> torch.Tensor:
    """Sliding-window mean confidence — the 'group confidence' DeepConf gates on.

    A single low-logprob token is noise; a *run* of them is a trace going off the
    rails. Averaging over a window of ``window`` tokens is the smoothing DeepConf
    uses so the early-abort floor triggers on sustained low confidence, not a
    one-token blip. Returns the per-position windowed mean (causal: position t
    averages the last ``window`` tokens up to and including t).
    """
    if conf.numel() == 0:
        return conf
    w = max(1, min(window, conf.numel()))
    # Causal moving average via cumulative sum (cheap, no conv kernels).
    csum = torch.cumsum(conf, dim=0)
    out = torch.empty_like(conf)
    for t in range(conf.numel()):
        lo = max(0, t - w + 1)
        total = csum[t] - (csum[lo - 1] if lo > 0 else 0.0)
        out[t] = total / (t - lo + 1)
    return out


def lowest_group_confidence(conf: torch.Tensor, window: int) -> float:
    """Trace-level score = the *minimum* sliding-window confidence along the trace.

    DeepConf scores a whole trace by its weakest stretch (the bottleneck), not
    its average — a trace that was confident everywhere except one fatal wobble
    should still be distrusted. Used both to weight the offline vote and as the
    online early-abort signal.
    """
    if conf.numel() == 0:
        return float("-inf")
    return float(sliding_group_confidence(conf, window).min())


def confidence_weighted_vote(
    answers: list, weights: list[float] | None = None
) -> tuple[object, dict]:
    """Aggregate trace answers into a single prediction.

    ``weights=None`` → plain majority vote (the self-consistency baseline).
    ``weights`` given → **confidence-weighted** vote: each trace contributes its
    confidence (mapped to a positive weight) to its answer's tally, so a handful
    of confident traces can outvote a noisy majority. Returns (winner, tally).
    """
    tally: dict = {}
    if weights is None:
        weights = [1.0] * len(answers)
    for ans, w in zip(answers, weights):
        tally[ans] = tally.get(ans, 0.0) + w
    if not tally:
        return None, tally
    winner = max(tally.items(), key=lambda kv: kv[1])[0]
    return winner, tally


def softmax_weights(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Map trace confidence scores to positive vote weights via a softmax.

    Confidences are logprobs (≤ 0 and unbounded below); a softmax turns them into
    a well-behaved positive distribution where more-confident traces get more
    say, with ``temperature`` controlling how peaked the weighting is.
    """
    if not scores:
        return []
    t = torch.tensor(scores, dtype=torch.float32) / max(temperature, 1e-6)
    return F.softmax(t, dim=-1).tolist()


def confidence_filtered_vote(
    answers: list, scores: list[float], *, keep_frac: float = 0.5
) -> tuple[object, dict]:
    """DeepConf's headline *offline* method: keep the top-``keep_frac`` traces by
    confidence, then take a plain majority among the survivors.

    Filtering (vs weighting) is what makes DeepConf beat self-consistency: a low
    keep-fraction discards the long tail of low-confidence (often wrong) traces
    entirely, so they can't dilute the vote at all — whereas confidence-*weighting*
    still lets every trace contribute a little. Keeps at least one trace. Returns
    (winner, tally over survivors)."""
    if not answers:
        return None, {}
    k = max(1, round(len(answers) * keep_frac))
    # Indices of the k highest-confidence traces.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    kept = [answers[i] for i in order]
    return confidence_weighted_vote(kept, weights=None)


# --------------------------------------------------------------------------- #
# Trace generation + the toy task.
# --------------------------------------------------------------------------- #
@dataclass
class Trace:
    tokens: list[int]
    answer: object
    confidence: float                       # lowest-group-confidence score
    aborted: bool = False
    emitted: int = 0                         # tokens actually generated
    group_curve: list[float] = field(default_factory=list)


@torch.no_grad()
def sample_trace(
    model: GPT,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    window: int,
    temperature: float,
    answer_fn,
    floor: float | None = None,
    generator: torch.Generator | None = None,
) -> Trace:
    """Sample one reasoning trace, tracking sliding-window confidence.

    If ``floor`` is given (online mode), abort as soon as the running
    sliding-window confidence drops below it — that's the token saving. The
    trace's score is the minimum windowed confidence seen (DeepConf's bottleneck
    score). ``answer_fn(tokens)`` extracts the trace's final answer.
    """
    block = model.cfg.block_size
    idx = prompt.clone()
    confs: list[float] = []
    group_curve: list[float] = []
    aborted = False
    for _ in range(max_new_tokens):
        logits = model.lm_head(model.hidden(idx[:, -block:]))[:, -1, :]  # [1, V]
        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits.float(), dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=generator)  # [1,1]
        conf = token_confidences(logits[0].unsqueeze(0), nxt[0]).item()
        confs.append(conf)
        # Running causal windowed mean of the last `window` confidences.
        w = confs[-window:]
        gmean = sum(w) / len(w)
        group_curve.append(gmean)
        idx = torch.cat([idx, nxt], dim=1)
        if floor is not None and gmean < floor:
            aborted = True
            break
    gen = idx[0, prompt.size(1):].tolist()
    conf_t = torch.tensor(confs)
    return Trace(
        tokens=gen,
        answer=answer_fn(gen),
        confidence=lowest_group_confidence(conf_t, window),
        aborted=aborted,
        emitted=len(gen),
        group_curve=group_curve,
    )


def _load_ckpt(path: str, device: str):
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = sd["cfg"]
    mcfg = GPTConfig(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_kv_head=cfg["n_kv_head"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"], dropout=0.0,
        rope_base=cfg["rope_base"],
        qk_norm=cfg.get("qk_norm", False),
        zero_init_proj=cfg.get("zero_init_proj", False),
        tie_embeddings=cfg.get("tie_embeddings", True),
        mtp_tokens=cfg.get("mtp_tokens", 0),
    )
    model = GPT(mcfg).to(device).eval()
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)
    return model, sd.get("meta", {})


def _random_model(device: str):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=128, n_layer=4, n_head=4, n_kv_head=4,
                    d_model=128, d_ffn=256)
    return GPT(cfg).to(device).eval(), {}


def _answer_last_token(tokens: list[int]) -> object:
    """Toy 'answer extractor': the last generated token mod 4.

    Stands in for parsing a final answer out of a reasoning trace. The exact map
    is irrelevant — what matters is that confidence correlates with which bucket
    a trace lands in, so the weighted vote can be exercised end-to-end.
    """
    return (tokens[-1] % 4) if tokens else None


# --------------------------------------------------------------------------- #
# Verifiable ADDITION task — the substrate for the real accuracy-lift number.
# --------------------------------------------------------------------------- #
def make_addition_answer_fn(meta: dict):
    """Build an answer-extractor that parses the integer a trace emits after '='.

    The model is trained on lines ``a+b=c\\n`` (see prepare_addition.py); a trace
    is the continuation after the ``a+b=`` prompt, so the answer is the digits up
    to the newline. ``reverse`` (LSB-first) answers are un-reversed. Returns
    ``None`` if the trace doesn't parse as an integer (counts as wrong)."""
    itos = meta["stoi"] and meta["itos"]
    reverse = bool(meta.get("reverse", False))

    def _fn(tokens: list[int]) -> object:
        s = "".join(itos[int(t)] for t in tokens)
        s = s.split("\n", 1)[0].strip()          # answer ends at the newline
        if reverse:
            s = s[::-1]
        try:
            return int(s)
        except ValueError:
            return None

    return _fn


def _run_addition_eval(model, meta, args, device):
    """Aggregate DeepConf measurement on held-out addition problems.

    For each of ``args.problems`` random ``a+b`` problems (ground truth known),
    sample ``args.traces`` traces and compare three deciders:
      * **single-sample** accuracy (the first trace — plain greedy-ish decode),
      * **majority vote** over k traces (self-consistency baseline),
      * **DeepConf confidence-weighted vote** over the same k traces,
    then, in a second pass with online early-abort, the **token savings** at the
    confidence-vote accuracy. This is the real accuracy-lift number — every
    answer is checked against ``a + b``.
    """
    import random

    stoi = meta["stoi"]
    rng = random.Random(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    answer_fn = make_addition_answer_fn(meta)
    max_val = 10 ** int(meta.get("digits", 2)) - 1

    single_ok = maj_ok = conf_ok = 0
    filt_ok = 0
    online_ok = 0
    online_tokens = full_tokens = 0
    conf_correct: list[float] = []
    conf_wrong: list[float] = []

    # Calibrate the online abort floor once, from a warmup set of traces.
    floor = None
    if args.online:
        warm_curves: list[float] = []
        for _ in range(args.warmup_traces):
            a, b = rng.randint(0, max_val), rng.randint(0, max_val)
            prompt = torch.tensor([[stoi[c] for c in f"{a}+{b}="]], device=device)
            tr = sample_trace(model, prompt, max_new_tokens=args.tokens, window=args.window,
                              temperature=args.temperature, answer_fn=answer_fn,
                              floor=None, generator=gen)
            warm_curves.extend(tr.group_curve)
        if warm_curves:
            floor = float(torch.quantile(torch.tensor(warm_curves), args.floor_percentile / 100.0))

    for _ in range(args.problems):
        a, b = rng.randint(0, max_val), rng.randint(0, max_val)
        truth = a + b
        prompt = torch.tensor([[stoi[c] for c in f"{a}+{b}="]], device=device)

        # Offline: k full traces -> majority vs confidence-weighted vote.
        traces = [
            sample_trace(model, prompt, max_new_tokens=args.tokens, window=args.window,
                         temperature=args.temperature, answer_fn=answer_fn,
                         floor=None, generator=gen)
            for _ in range(args.traces)
        ]
        answers = [t.answer for t in traces]
        scores = [t.confidence for t in traces]
        full_tokens += sum(t.emitted for t in traces)

        single_ok += int(answers[0] == truth)
        maj, _ = confidence_weighted_vote(answers, weights=None)
        maj_ok += int(maj == truth)
        weights = softmax_weights(scores, temperature=args.vote_temp)
        conf_winner, _ = confidence_weighted_vote(answers, weights=weights)
        conf_ok += int(conf_winner == truth)
        # DeepConf's headline offline method: filter to the top-keep_frac, vote.
        filt_winner, _ = confidence_filtered_vote(answers, scores, keep_frac=args.keep_frac)
        filt_ok += int(filt_winner == truth)

        # Track confidence vs correctness (the load-bearing correlation).
        for ans, sc in zip(answers, scores):
            (conf_correct if ans == truth else conf_wrong).append(sc)

        # Online: early-abort traces, then confidence-weighted vote on survivors.
        if args.online:
            otr = [
                sample_trace(model, prompt, max_new_tokens=args.tokens, window=args.window,
                             temperature=args.temperature, answer_fn=answer_fn,
                             floor=floor, generator=gen)
                for _ in range(args.traces)
            ]
            online_tokens += sum(t.emitted for t in otr)
            o_ans = [t.answer for t in otr]
            o_w = softmax_weights([t.confidence for t in otr], temperature=args.vote_temp)
            o_winner, _ = confidence_weighted_vote(o_ans, weights=o_w)
            online_ok += int(o_winner == truth)

    n = args.problems
    print(f"\n  problems={n}  traces/problem={args.traces}  temperature={args.temperature}")
    print("\n  decider                         accuracy        tokens     vs full")
    print(f"  single sample (1 trace)         {single_ok / n:7.1%}        "
          f"{full_tokens // args.traces:>6}     {1 / args.traces:4.2f}x")
    print(f"  majority vote (k={args.traces:<2})            {maj_ok / n:7.1%}        "
          f"{full_tokens:>6}     1.00x")
    print(f"  DeepConf conf-weighted vote     {conf_ok / n:7.1%}        "
          f"{full_tokens:>6}     1.00x")
    print(f"  DeepConf conf-filtered (top {int(args.keep_frac * 100):>2}%)  {filt_ok / n:7.1%}        "
          f"{full_tokens:>6}     1.00x")
    if args.online:
        print(f"  DeepConf online (early-abort)   {online_ok / n:7.1%}        "
              f"{online_tokens:>6}     {online_tokens / full_tokens:4.2f}x")
        print(f"\n  online token savings:           {full_tokens} -> {online_tokens} "
              f"({(1 - online_tokens / full_tokens) * 100:.0f}% fewer)")
    # The correlation that makes the weighting work at all.
    if conf_correct and conf_wrong:
        mc = sum(conf_correct) / len(conf_correct)
        mw = sum(conf_wrong) / len(conf_wrong)
        print(f"\n  mean confidence  correct={mc:.3f}  wrong={mw:.3f}  "
              f"(gap {mc - mw:+.3f} — higher ⇒ confidence tracks correctness)")
    print(f"  accuracy lift (conf vote − majority):  {(conf_ok - maj_ok) / n:+.1%}")
    print(f"  accuracy lift (conf filter − majority): {(filt_ok - maj_ok) / n:+.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint path (omit for a random tiny model)")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--task", default="freeform", choices=["freeform", "add"],
                    help="'add' runs the verifiable addition accuracy-lift eval; "
                         "'freeform' runs the single-prompt mechanism demo")
    ap.add_argument("--problems", type=int, default=200,
                    help="(--task add) number of held-out problems to average over")
    ap.add_argument("--traces", type=int, default=16, help="number of parallel traces (k)")
    ap.add_argument("--tokens", type=int, default=64, help="max new tokens per trace")
    ap.add_argument("--window", type=int, default=8, help="sliding window for group confidence")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--vote-temp", type=float, default=1.0,
                    help="softmax temperature for the confidence->weight mapping")
    ap.add_argument("--keep-frac", type=float, default=0.5,
                    help="(--task add) fraction of most-confident traces the "
                         "confidence-FILTERED vote keeps before majority")
    ap.add_argument("--online", action="store_true", help="enable online early-abort")
    ap.add_argument("--warmup-traces", type=int, default=4,
                    help="traces used to calibrate the early-abort floor (online)")
    ap.add_argument("--floor-percentile", type=float, default=10.0,
                    help="abort floor = this percentile of warmup group-confidence")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.ckpt:
        model, meta = _load_ckpt(args.ckpt, device)
        src = args.ckpt
    else:
        model, meta = _random_model(device)
        src = "<random tiny model>"

    # Verifiable addition eval: aggregate accuracy lift over held-out problems.
    if args.task == "add":
        if "stoi" not in meta:
            raise SystemExit(
                "--task add needs an addition checkpoint (meta with stoi/itos). "
                "Train one: python prepare_addition.py && python train.py --config configs/tiny_add.py"
            )
        print(f"[bench] model={src}")
        print(f"[bench] device={device}  task=add  online={args.online}")
        _run_addition_eval(model, meta, args, device)
        return

    # Encode the prompt (same scheme as the MTP bench).
    if meta.get("tokenizer") == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode(args.prompt) or [enc.eot_token]
    elif "stoi" in meta:
        ids = [meta["stoi"].get(c, 0) for c in args.prompt] or [0]
    else:
        ids = [1, 2, 3, 4]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[bench] model={src}")
    print(f"[bench] device={device}  traces={args.traces}  tokens={args.tokens}  "
          f"window={args.window}  online={args.online}")

    # ---- Optional online calibration: a small warmup sets the abort floor. ----
    floor = None
    if args.online:
        warm = [
            sample_trace(model, prompt, max_new_tokens=args.tokens, window=args.window,
                         temperature=args.temperature, answer_fn=_answer_last_token,
                         floor=None, generator=gen)
            for _ in range(args.warmup_traces)
        ]
        curve = torch.tensor([g for w in warm for g in w.group_curve])
        floor = float(torch.quantile(curve, args.floor_percentile / 100.0)) if curve.numel() else None
        print(f"[bench] online abort floor (p{args.floor_percentile:.0f} of warmup "
              f"group-confidence): {floor:.3f}")

    # ---- Generate the k traces (offline: floor=None; online: with floor). ----
    traces = [
        sample_trace(model, prompt, max_new_tokens=args.tokens, window=args.window,
                     temperature=args.temperature, answer_fn=_answer_last_token,
                     floor=floor, generator=gen)
        for _ in range(args.traces)
    ]
    total_tokens = sum(t.emitted for t in traces)
    full_tokens = args.traces * args.tokens
    n_aborted = sum(t.aborted for t in traces)

    answers = [t.answer for t in traces]
    scores = [t.confidence for t in traces]

    # Baseline: plain majority vote (self-consistency).
    maj, maj_tally = confidence_weighted_vote(answers, weights=None)
    # DeepConf: confidence-weighted vote.
    weights = softmax_weights(scores, temperature=1.0)
    conf_winner, conf_tally = confidence_weighted_vote(answers, weights=weights)

    print("\n  strategy                 winner    tokens     vs full")
    print(f"  majority vote            {str(maj):>6}    {full_tokens:>6}     1.00x")
    label = "DeepConf (online)" if args.online else "DeepConf (offline)"
    print(f"  {label:<22}   {str(conf_winner):>6}    {total_tokens:>6}     "
          f"{total_tokens / full_tokens:4.2f}x")
    if args.online:
        print(f"\n  traces aborted early:    {n_aborted}/{args.traces}")
        print(f"  token savings:           {full_tokens} -> {total_tokens} "
              f"({(1 - total_tokens / full_tokens) * 100:.0f}% fewer)")
    print(f"  answer distribution:     {dict(Counter(answers))}")
    print(f"  confidence range:        [{min(scores):.2f}, {max(scores):.2f}]")

    if not args.ckpt:
        print("\n[note] random untrained model: traces are ~uniform so the vote is "
              "near-random. Point --ckpt at a trained reasoning checkpoint to see the "
              "offline accuracy lift; the online token savings are model-agnostic.")


if __name__ == "__main__":
    main()

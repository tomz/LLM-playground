"""DeepConf confidence-filtering helpers (tools/bench_deepconf.py).

Pins the *mechanism* on CPU with no model: per-token confidence, sliding-window
group confidence, the bottleneck (min-group) trace score, and that the
confidence-weighted vote can outvote a noisy plain majority. Also runs the
end-to-end bench on a tiny random model to prove online early-abort never emits
*more* tokens than the offline (full-length) pass.
"""
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_deepconf import (  # noqa: E402
    confidence_weighted_vote,
    confidence_filtered_vote,
    lowest_group_confidence,
    sample_trace,
    sliding_group_confidence,
    softmax_weights,
    token_confidences,
    _answer_last_token,
    _random_model,
)
from model import GPT, GPTConfig  # noqa: E402


def test_token_confidence_is_logprob_of_sampled_token():
    # Two steps, vocab 4. A confident step (peaked) vs a flat step.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0],     # very peaked on token 0
                           [0.0, 0.0, 0.0, 0.0]])     # uniform
    ids = torch.tensor([0, 2])
    conf = token_confidences(logits, ids)
    assert conf.shape == (2,)
    # Peaked-correct token → confidence near 0 (logprob ~0); uniform → log(1/4).
    assert conf[0] > -0.01
    assert abs(conf[1].item() - torch.log(torch.tensor(0.25)).item()) < 1e-5
    # Confidence is always a logprob ≤ 0.
    assert torch.all(conf <= 1e-6)


def test_sliding_group_confidence_smooths_single_dips():
    # A lone low-confidence token shouldn't tank the windowed mean as hard as a
    # sustained run of them does.
    conf = torch.tensor([0.0, 0.0, -6.0, 0.0, 0.0])
    g = sliding_group_confidence(conf, window=3)
    assert g.shape == conf.shape
    # The windowed min is higher (less alarming) than the raw single-token min.
    assert g.min() > conf.min()


def test_lowest_group_confidence_is_the_bottleneck():
    # A trace confident everywhere except a sustained bad stretch scores by that
    # stretch (the minimum windowed mean), not by its overall average.
    good = torch.zeros(20)
    bad = torch.zeros(20)
    bad[8:12] = -8.0                               # a sustained wobble
    assert lowest_group_confidence(bad, window=4) < lowest_group_confidence(good, window=4)
    # Empty trace → -inf sentinel (least trustworthy).
    assert lowest_group_confidence(torch.empty(0), window=4) == float("-inf")


def test_majority_vote_is_the_unweighted_baseline():
    answers = ["a", "a", "b"]
    winner, tally = confidence_weighted_vote(answers, weights=None)
    assert winner == "a"
    assert tally == {"a": 2.0, "b": 1.0}


def test_confidence_weighting_can_overturn_a_noisy_majority():
    # Three traces say "wrong" but with low confidence; two say "right" loudly.
    answers = ["wrong", "wrong", "wrong", "right", "right"]
    scores = [-9.0, -9.0, -9.0, -0.1, -0.1]        # logprob-style confidences
    weights = softmax_weights(scores, temperature=1.0)
    maj, _ = confidence_weighted_vote(answers, weights=None)
    conf_winner, _ = confidence_weighted_vote(answers, weights=weights)
    assert maj == "wrong"                          # plain majority is fooled
    assert conf_winner == "right"                  # confidence weighting recovers


def test_softmax_weights_normalise_and_rank():
    w = softmax_weights([-1.0, -1.0, 0.0])
    assert abs(sum(w) - 1.0) < 1e-6
    assert w[2] > w[0] and w[2] > w[1]             # the confident trace weighs more
    assert softmax_weights([]) == []


def test_confidence_filtered_vote_drops_low_confidence_tail():
    # 5 traces: a low-confidence majority says "wrong", a high-confidence pair
    # says "right". Keeping the top 40% (2 traces) votes only among the
    # confident ones -> "right". This is DeepConf's headline offline method.
    answers = ["wrong", "wrong", "wrong", "right", "right"]
    scores = [-9.0, -8.5, -8.0, -0.2, -0.1]
    winner, tally = confidence_filtered_vote(answers, scores, keep_frac=0.4)
    assert winner == "right"
    assert set(tally) == {"right"}                 # only the kept (confident) traces count


def test_confidence_filtered_vote_keeps_at_least_one():
    # Even an absurdly small keep_frac must keep the single most-confident trace.
    answers = ["a", "b", "c"]
    scores = [-5.0, -1.0, -9.0]
    winner, _ = confidence_filtered_vote(answers, scores, keep_frac=0.0)
    assert winner == "b"                           # the top-confidence trace


def test_confidence_filtered_vote_full_keep_is_plain_majority():
    # keep_frac=1.0 keeps everything -> identical to a plain majority vote.
    answers = ["a", "a", "b"]
    scores = [-3.0, -2.0, -0.1]                    # confidence can't override at full keep
    winner, _ = confidence_filtered_vote(answers, scores, keep_frac=1.0)
    maj, _ = confidence_weighted_vote(answers, weights=None)
    assert winner == maj == "a"


def test_online_early_abort_never_emits_more_than_offline():
    device = "cpu"
    model, _ = _random_model(device)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(0)

    offline = sample_trace(model, prompt, max_new_tokens=24, window=4, temperature=1.0,
                           answer_fn=_answer_last_token, floor=None, generator=gen)
    # A very high floor forces aggressive aborting; it must cut tokens, never add.
    online = sample_trace(model, prompt, max_new_tokens=24, window=4, temperature=1.0,
                          answer_fn=_answer_last_token, floor=10.0, generator=gen)
    assert offline.emitted == 24                   # offline runs to the budget
    assert online.emitted <= offline.emitted       # online can only save tokens
    assert online.aborted                          # floor=+10 (above any logprob) aborts immediately
    # An aborted trace still yields a (possibly weaker) answer + a finite-ish score.
    assert online.answer is not None


def test_sample_trace_is_deterministic_under_a_seeded_generator():
    device = "cpu"
    model, _ = _random_model(device)
    prompt = torch.tensor([[5, 6, 7]], dtype=torch.long, device=device)
    a = sample_trace(model, prompt, max_new_tokens=16, window=4, temperature=1.0,
                     answer_fn=_answer_last_token,
                     generator=torch.Generator(device=device).manual_seed(42))
    b = sample_trace(model, prompt, max_new_tokens=16, window=4, temperature=1.0,
                     answer_fn=_answer_last_token,
                     generator=torch.Generator(device=device).manual_seed(42))
    assert a.tokens == b.tokens
    assert a.confidence == b.confidence

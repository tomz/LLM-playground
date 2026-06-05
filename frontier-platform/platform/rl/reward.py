"""Reward shaping for RLVR (see docs/15-reasoning-rl-rlvr.md §"Stability").

A raw verifier (``platform.rl.verifiers``) answers "is the final answer correct?"
That signal alone is enough to *learn*, but bare correctness reward is brittle:

  - **Reward hacking:** the model finds degenerate ways to trip the verifier
    (dumping the expected string, emitting many candidate answers, exploiting a
    lenient regex). Held-out checks and anti-gaming guards penalize these.
  - **Length explosion:** unbounded chain-of-thought; long answers cost serving
    money and often signal flailing. A soft length penalty curbs it.
  - **Format drift:** reasoning models are trained to think in a tagged format
    (e.g. ``<think>...</think>`` then a boxed/final answer). A small format
    reward keeps rollouts parseable so the verifier can find the answer.

``CompositeReward`` blends these into a single scalar the GRPO learner consumes,
and records a per-component breakdown for monitoring. It is deterministic and
cheap — the whole point of RLVR is that the reward is not a learned model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .verifiers import Verifier


# --- format / anti-hacking checks (all return reward in roughly [-1, 1]) ---

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{[^}]*\}|<answer>.*?</answer>", re.DOTALL)


def format_reward(response: str, *, reward: float = 0.2) -> float:
    """Small bonus for emitting the expected think→answer structure.

    Full bonus iff there is exactly one ``<think>...</think>`` block followed by a
    boxed/`<answer>` final answer. Partial credit for having one of the two.
    """
    has_think = len(_THINK_RE.findall(response)) == 1
    has_answer = _BOXED_RE.search(response) is not None
    return reward * (0.5 * has_think + 0.5 * has_answer)


def soft_length_penalty(response: str, *, target_tokens: int = 512,
                        max_tokens: int = 2048, coef: float = 0.1,
                        count_tokens=None) -> float:
    """Zero up to ``target_tokens``, then ramps linearly to ``-coef`` at
    ``max_tokens`` (and stays at ``-coef`` beyond).

    Token count comes from ``count_tokens(response) -> int`` when provided (pass
    ``tokenizer.encode`` for exact counts); otherwise it falls back to a
    whitespace-word approximation."""
    n = len(count_tokens(response)) if count_tokens is not None else len(response.split())
    if n <= target_tokens:
        return 0.0
    frac = min(1.0, (n - target_tokens) / max(1, max_tokens - target_tokens))
    return -coef * frac


def repetition_penalty(response: str, *, ngram: int = 3, coef: float = 0.5) -> float:
    """Penalize degenerate loops (a classic reward-hack / collapse symptom).

    Returns ``-coef * (1 - distinct_ngram_ratio)``; 0 for non-repetitive text.
    """
    toks = response.split()
    if len(toks) < ngram + 1:
        return 0.0
    grams = [tuple(toks[i:i + ngram]) for i in range(len(toks) - ngram + 1)]
    if not grams:
        return 0.0
    distinct_ratio = len(set(grams)) / len(grams)
    return -coef * (1.0 - distinct_ratio)


def answer_spam_guard(response: str, *, max_candidates: int = 3, coef: float = 1.0) -> float:
    """Anti-hack: penalize "shotgun" answers that list many numbers/boxes hoping
    one matches the verifier's last-number / boxed extraction. Returns ``-coef``
    when more than ``max_candidates`` boxed/answer spans are present."""
    n_boxed = len(_BOXED_RE.findall(response))
    return -coef if n_boxed > max_candidates else 0.0


def language_consistency_reward(response: str, *, target_lang: str = "en",
                                coef: float = 0.5, min_conf: float = 0.0) -> float:
    """Penalize responses that drift out of the target language.

    Reasoning models trained with RLVR are prone to *language mixing* — the CoT
    silently code-switches (e.g. English prompt, half-Chinese scratchpad), which
    DeepSeek-R1 explicitly added a language-consistency reward to suppress. We
    reuse the data pipeline's :func:`platform.data.filter.detect_language` LID:
    reward ``0`` when the detected language matches ``target_lang``, else
    ``-coef``. The ``<think>`` tags and boxed-answer markup are stripped before
    detection so formatting tokens don't skew the LID. When the detector is
    unsure (confidence < ``min_conf``, or no alphabetic content) the reward is
    ``0`` — we only penalize *confident* drift, never short/numeric answers."""
    from ..data.filter import detect_language

    # Strip think/answer markup so LID sees prose, not tags.
    text = _THINK_RE.sub(" ", response)
    text = _BOXED_RE.sub(" ", text)
    lang, conf = detect_language(text)
    if lang == "unk" or conf < min_conf:
        return 0.0
    return 0.0 if lang == target_lang else -coef


@dataclass
class RewardConfig:
    correctness_weight: float = 1.0
    format_weight: float = 1.0
    length_target_tokens: int = 512
    length_max_tokens: int = 2048
    length_coef: float = 0.1
    # Difficulty-aware length budget. When a per-prompt difficulty in [0,1] is
    # available, the length target scales between the easy and hard budgets so a
    # hard problem is allowed a longer chain-of-thought before the soft penalty
    # engages (easy problems are held short to curb rambling). Set
    # difficulty_aware=False to use the flat length_target_tokens for every prompt.
    difficulty_aware: bool = False
    length_target_easy: int = 256
    length_target_hard: int = 1536
    repetition_coef: float = 0.5
    answer_spam_max: int = 3
    answer_spam_coef: float = 1.0
    # Language-consistency shaping (DeepSeek-R1-style anti language-mixing).
    language_weight: float = 0.0      # 0 disables; >0 enables the penalty
    target_lang: str = "en"
    language_min_conf: float = 0.0
    # Clip the final shaped reward into a bounded range (keeps GRPO advantages sane).
    clip: tuple[float, float] = (-2.0, 2.0)

    def length_target_for(self, difficulty: float | None) -> int:
        """Resolve the soft-length target for a prompt of the given difficulty.

        ``difficulty`` is in ``[0, 1]`` (0 = easiest, 1 = hardest). Returns the
        flat ``length_target_tokens`` when difficulty-awareness is off or no
        difficulty is supplied; otherwise linearly interpolates between
        ``length_target_easy`` and ``length_target_hard``."""
        if not self.difficulty_aware or difficulty is None:
            return self.length_target_tokens
        d = max(0.0, min(1.0, float(difficulty)))
        lo, hi = self.length_target_easy, self.length_target_hard
        return int(round(lo + (hi - lo) * d))


@dataclass
class CompositeReward:
    """Blend a correctness ``Verifier`` with format/length/anti-hacking shaping.

    Callable as a ``Verifier`` (``(prompt, response) -> float``) so it drops
    straight into ``run_grpo``. Use :meth:`breakdown` for per-component logging.

    ``difficulty_fn`` (optional) maps a *prompt* to a difficulty in ``[0, 1]``;
    when set together with ``RewardConfig.difficulty_aware`` the soft-length
    budget scales per prompt (hard problems may reason longer before the penalty
    bites). A real run sources difficulty from the prompt's metadata (pass-rate,
    grade level, solver depth); for tests any ``prompt -> float`` works.
    """

    verifier: Verifier
    cfg: RewardConfig = field(default_factory=RewardConfig)
    tokenizer: object | None = None   # if set, length penalty uses real token counts
    difficulty_fn: Callable[[str], float] | None = None

    @property
    def count_tokens(self):
        tok = self.tokenizer
        if tok is not None and hasattr(tok, "encode"):
            return tok.encode
        return None

    def breakdown(self, prompt: str, response: str) -> dict[str, float]:
        c = self.cfg
        correctness = c.correctness_weight * self.verifier(prompt, response)
        fmt = c.format_weight * format_reward(response)
        difficulty = self.difficulty_fn(prompt) if self.difficulty_fn is not None else None
        length = soft_length_penalty(
            response, target_tokens=c.length_target_for(difficulty),
            max_tokens=c.length_max_tokens, coef=c.length_coef,
            count_tokens=self.count_tokens,
        )
        rep = repetition_penalty(response, coef=c.repetition_coef)
        spam = answer_spam_guard(
            response, max_candidates=c.answer_spam_max, coef=c.answer_spam_coef
        )
        lang = 0.0
        if c.language_weight > 0.0:
            lang = c.language_weight * language_consistency_reward(
                response, target_lang=c.target_lang, coef=1.0,
                min_conf=c.language_min_conf,
            )
        total = correctness + fmt + length + rep + spam + lang
        lo, hi = c.clip
        total = max(lo, min(hi, total))
        return {
            "correctness": correctness, "format": fmt, "length": length,
            "repetition": rep, "answer_spam": spam, "language": lang,
            "total": total,
        }

    def __call__(self, prompt: str, response: str) -> float:
        return self.breakdown(prompt, response)["total"]

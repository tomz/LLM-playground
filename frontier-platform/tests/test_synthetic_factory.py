"""Tests for the synthetic data factory.

The factory replaces what used to be a 16-line word-bag generator. These tests
exercise the four extension points (teachers, policies, verifiers, dedup +
decontamination) plus lineage round-tripping and JSONL output.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform.data.dedup import MinHashDeduper
from platform.data.decontaminate import Decontaminator
from platform.data.synthetic import (
    CallableTeacher,
    EchoTeacher,
    FactoryStats,
    MathProblemPolicy,
    QAPolicy,
    ReasoningTracePolicy,
    RephrasePolicy,
    SampleRecord,
    SyntheticFactory,
    TemplatePolicy,
    TemplateTeacher,
    TextbookPolicy,
    read_lineage_jsonl,
    write_corpus,
    write_lineage_jsonl,
)


# ----- back-compat ----------------------------------------------------------


def test_write_corpus_back_compat_shape(tmp_path):
    """The shim must still produce a 20-file deterministic corpus for tests."""
    root = write_corpus(tmp_path / "c", n_files=20, words_per_file=200, seed=0)
    files = sorted(root.glob("doc_*.txt"))
    assert len(files) == 20
    txt = files[0].read_text(encoding="utf-8")
    assert txt.endswith(".\n")
    assert len(txt.split()) >= 200


def test_write_corpus_seed_is_deterministic(tmp_path):
    a = write_corpus(tmp_path / "a", n_files=3, seed=42)
    b = write_corpus(tmp_path / "b", n_files=3, seed=42)
    for f1, f2 in zip(sorted(a.glob("*.txt")), sorted(b.glob("*.txt"))):
        assert f1.read_text() == f2.read_text()


# ----- teachers -------------------------------------------------------------


def test_echo_teacher_returns_prompt():
    t = EchoTeacher()
    assert t.generate("hello") == "hello"
    assert t.name == "echo"


def test_template_teacher_substitutes():
    t = TemplateTeacher(template="A: {prompt}!", name="ans")
    assert t.generate("ok") == "A: ok!"
    assert t.name == "ans"


def test_callable_teacher_wraps_lambda():
    t = CallableTeacher(fn=lambda p: p.upper(), name="upper")
    assert t.generate("hi") == "HI"


# ----- policies -------------------------------------------------------------


def test_template_policy_emits_n_prompts():
    pol = TemplatePolicy(
        templates=["What about {x}?", "Tell me about {x}."],
        slot_values={"x": ["cats", "dogs"]},
    )
    import random
    rng = random.Random(0)
    prompts = list(pol.prompts(8, rng=rng))
    assert len(prompts) == 8
    assert all("cats" in p or "dogs" in p for p in prompts)
    assert pol.acceptance_verifier() is None


def test_textbook_policy_uses_topic_and_audience():
    pol = TextbookPolicy()
    import random
    rng = random.Random(1)
    prompts = list(pol.prompts(3, rng=rng))
    assert all("Explain" in p and "three short paragraphs" in p for p in prompts)


def test_math_problem_policy_with_fixed_pairs_has_known_answers():
    pol = MathProblemPolicy(fixed=[("What is 2 + 3?", 5), ("What is 7 + 1?", 8)])
    import random
    prompts = list(pol.prompts(4, rng=random.Random(0)))
    # Fixed pairs cycle.
    assert prompts[0] == "What is 2 + 3?"
    assert prompts[2] == "What is 2 + 3?"
    v = pol.acceptance_verifier()
    assert v is not None
    assert v("What is 2 + 3?", r"The answer is \boxed{5}.") == 1.0
    assert v("What is 2 + 3?", r"The answer is \boxed{7}.") == 0.0


def test_math_problem_policy_random_arithmetic_round_trips():
    pol = MathProblemPolicy(n_max=10, op="+")
    import random
    rng = random.Random(7)
    prompts = list(pol.prompts(5, rng=rng))
    v = pol.acceptance_verifier()
    # Verifier knows the answer because the policy stamped it.
    for p in prompts:
        # The prompt looks like "What is A + B? Reply with the number inside \boxed{}."
        head = p.split("?", 1)[0]
        a, b = [int(t) for t in head.replace("What is", "").split("+")]
        assert v(p, rf"thinking... \boxed{{{a + b}}}") == 1.0
        assert v(p, rf"thinking... \boxed{{{a + b + 1}}}") == 0.0


def test_reasoning_trace_policy_accepts_correct_final_answer():
    pol = ReasoningTracePolicy(problems=[("Compute 6 * 7.", 42)])
    import random
    prompts = list(pol.prompts(2, rng=random.Random(0)))
    v = pol.acceptance_verifier()
    assert all("Solve the problem" in p for p in prompts)
    good = "First multiply 6 by 7, which is 42. " r"So \boxed{42}."
    bad = "First multiply 6 by 7, which is 43. " r"So \boxed{43}."
    assert v(prompts[0], good) == 1.0
    assert v(prompts[0], bad) == 0.0


def test_qa_and_rephrase_policies_emit_passage_aware_prompts():
    passages = ["The Krebs cycle is a series of chemical reactions.",
                "Photosynthesis turns sunlight into glucose."]
    rng_qa = __import__("random").Random(0)
    qa = QAPolicy(passages=passages, n_questions=2)
    prompts = list(qa.prompts(3, rng=rng_qa))
    assert all(any(pa in p for pa in passages) for p in prompts)
    assert all("Q:" in p and "A:" in p for p in prompts)

    rng_re = __import__("random").Random(0)
    rp = RephrasePolicy(seed_passages=passages)
    prompts = list(rp.prompts(3, rng=rng_re))
    assert all("Rewrite" in p for p in prompts)


# ----- factory: end-to-end --------------------------------------------------


def test_factory_yields_n_accepted_with_no_verifier():
    teacher = TemplateTeacher(template="explanation of {prompt}", name="t")
    policy = TextbookPolicy()
    fac = SyntheticFactory(teacher=teacher, policy=policy, seed=0)
    out = list(fac.generate(5))
    assert len(out) == 5
    assert all(r.accepted for r in out)
    assert all(isinstance(r, SampleRecord) for r in out)
    assert fac.stats.accepted == 5
    assert fac.stats.attempted == 5
    assert fac.stats.acceptance_rate() == 1.0
    # All records carry the teacher + policy names and a deterministic id.
    assert {r.teacher for r in out} == {"t"}
    assert {r.policy for r in out} == {"textbook"}


def test_factory_rejection_sampler_filters_wrong_answers():
    """MathProblemPolicy: teacher echoing the prompt fails the verifier."""
    fixed = [("What is 1 + 1?", 2)]
    pol = MathProblemPolicy(fixed=fixed)
    teacher = EchoTeacher()   # never produces \boxed{2}
    fac = SyntheticFactory(teacher=teacher, policy=pol, seed=0,
                           max_attempts_per_sample=4, emit_rejections=True)
    out = list(fac.generate(3))
    accepted = [r for r in out if r.accepted]
    rejected = [r for r in out if not r.accepted]
    assert len(accepted) == 0
    # Every attempt should be rejected with reason='verifier'.
    assert rejected, "expected some rejections to be emitted"
    assert all(r.rejection_reason == "verifier" for r in rejected)
    assert fac.stats.rejected_verifier == len(rejected)


def test_factory_rejection_sampler_accepts_correct_answers():
    """Same policy with a teacher that gets it right: 100% acceptance."""
    fixed = [("What is 1 + 1?", 2)]
    pol = MathProblemPolicy(fixed=fixed)
    teacher = CallableTeacher(fn=lambda _: r"\boxed{2}", name="oracle")
    fac = SyntheticFactory(teacher=teacher, policy=pol, seed=0)
    out = list(fac.generate(3))
    assert len(out) == 3
    assert all(r.accepted for r in out)
    assert all(r.verifier_score == 1.0 for r in out)


def test_factory_dedup_drops_repeated_responses():
    """A teacher that always returns the same string should produce 1 sample
    even when we ask for many, because the deduper rejects every repeat."""
    teacher = TemplateTeacher(template="constant response", name="const")
    pol = TextbookPolicy()
    deduper = MinHashDeduper(num_perm=64, threshold=0.5, ngram=2)
    fac = SyntheticFactory(
        teacher=teacher, policy=pol, deduper=deduper, seed=0,
        max_attempts_per_sample=5, emit_rejections=True,
    )
    out = list(fac.generate(4))
    accepted = [r for r in out if r.accepted]
    rejected = [r for r in out if not r.accepted]
    assert len(accepted) == 1
    assert all(r.rejection_reason == "duplicate" for r in rejected)
    assert fac.stats.rejected_duplicate >= 1


def test_factory_decontamination_drops_eval_overlap(tmp_path):
    """Responses overlapping the eval index get rejected."""
    eval_file = tmp_path / "eval.txt"
    # The decontaminator is n=13 by default; give it enough material.
    eval_text = (
        "Photosynthesis is the process by which green plants use sunlight to "
        "synthesise foods with the help of chlorophyll and water and carbon "
        "dioxide producing glucose and releasing oxygen as a byproduct."
    )
    eval_file.write_text(eval_text)
    deco = Decontaminator([str(eval_file)], n=13)

    # Teacher returns the eval sentence verbatim => contaminated.
    teacher = TemplateTeacher(template=eval_text, name="leaky")
    pol = TextbookPolicy()
    fac = SyntheticFactory(
        teacher=teacher, policy=pol, decontaminator=deco, seed=0,
        max_attempts_per_sample=3, emit_rejections=True,
    )
    out = list(fac.generate(2))
    accepted = [r for r in out if r.accepted]
    rejected = [r for r in out if not r.accepted]
    assert accepted == []
    assert rejected, "expected contamination rejections"
    assert all(r.rejection_reason == "contaminated" for r in rejected)


def test_factory_empty_response_is_rejected():
    teacher = TemplateTeacher(template="", name="silent")
    pol = TextbookPolicy()
    fac = SyntheticFactory(
        teacher=teacher, policy=pol, seed=0,
        max_attempts_per_sample=2, emit_rejections=True,
    )
    out = list(fac.generate(2))
    assert all(not r.accepted for r in out)
    assert all(r.rejection_reason == "empty_response" for r in out)


def test_factory_sample_ids_are_deterministic_given_seed():
    teacher = TemplateTeacher(template="A: {prompt}", name="t")
    pol = TextbookPolicy()
    a = list(SyntheticFactory(teacher=teacher, policy=pol, seed=123).generate(3))
    b = list(SyntheticFactory(teacher=teacher, policy=pol, seed=123).generate(3))
    assert [r.sample_id for r in a] == [r.sample_id for r in b]
    assert [r.prompt for r in a] == [r.prompt for r in b]
    assert [r.response for r in a] == [r.response for r in b]


def test_factory_stats_accounting():
    """Mixed acceptance: half correct, half wrong, no other filters."""
    fixed = [("What is 2 + 2?", 4)]
    pol = MathProblemPolicy(fixed=fixed)
    calls = {"i": 0}

    def alternating(_):
        calls["i"] += 1
        return r"\boxed{4}" if calls["i"] % 2 == 0 else r"\boxed{99}"

    teacher = CallableTeacher(fn=alternating, name="alt")
    fac = SyntheticFactory(teacher=teacher, policy=pol, seed=0,
                           max_attempts_per_sample=4)
    out = list(fac.generate(3))
    assert len(out) == 3
    s: FactoryStats = fac.stats
    assert s.accepted == 3
    assert s.rejected_verifier >= 3   # at least one failure per accepted sample
    assert 0.0 < s.acceptance_rate() <= 1.0
    assert "rejected_verifier" in s.as_dict()


# ----- JSONL lineage --------------------------------------------------------


def test_write_lineage_jsonl_only_accepted(tmp_path):
    teacher = CallableTeacher(fn=lambda _: r"\boxed{4}", name="oracle")
    pol = MathProblemPolicy(fixed=[("What is 2 + 2?", 4)])
    fac = SyntheticFactory(teacher=teacher, policy=pol, seed=0)
    path = fac.write_jsonl(3, tmp_path / "out.jsonl", only_accepted=True)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for ln in lines:
        rec = json.loads(ln)
        assert rec["accepted"] is True
        assert rec["verifier_score"] == 1.0
        assert rec["schema_version"] == 1


def test_write_lineage_jsonl_full(tmp_path):
    """only_accepted=False writes accepted + rejected for audit."""
    teacher = EchoTeacher()
    pol = MathProblemPolicy(fixed=[("What is 1 + 1?", 2)])
    fac = SyntheticFactory(teacher=teacher, policy=pol, seed=0,
                           max_attempts_per_sample=3)
    # All will be rejected because echo can't say \boxed{2}.
    path = fac.write_jsonl(2, tmp_path / "full.jsonl", only_accepted=False)
    records = list(read_lineage_jsonl(path))
    assert records, "expected rejected records for audit"
    assert all(not r.accepted for r in records)
    assert all(r.rejection_reason == "verifier" for r in records)


def test_round_trip_through_read_lineage_jsonl(tmp_path):
    teacher = TemplateTeacher(template="resp {prompt}", name="t")
    pol = TextbookPolicy()
    fac = SyntheticFactory(teacher=teacher, policy=pol, seed=99)
    records = list(fac.generate(4))
    path = write_lineage_jsonl(records, tmp_path / "lin.jsonl")
    loaded = list(read_lineage_jsonl(path))
    assert len(loaded) == 4
    for a, b in zip(records, loaded):
        assert a.sample_id == b.sample_id
        assert a.prompt == b.prompt
        assert a.response == b.response
        assert a.accepted == b.accepted
        assert a.verifier_score == b.verifier_score


def test_factory_with_engine_teacher_smoke():
    """Smoke test the EngineTeacher adapter using a fake Engine + tokenizer.

    The factory should drive the engine through asyncio and collect tokens
    transparently."""
    pytest.importorskip("torch")

    class _FakeTokenizer:
        def encode(self, s: str) -> list[int]:
            return [ord(c) % 256 for c in s][:8]

        def decode(self, ids: list[int]) -> str:
            return "ok " + "".join(chr(i % 128) for i in ids[:4])

    class _FakeEngine:
        async def generate(self, req):
            for tok in [10, 20, 30]:
                yield {"token_id": tok, "logprob": -0.1, "text": None, "done": False}
            yield {"done": True, "usage": {"prompt_tokens": len(req.prompt_ids),
                                            "completion_tokens": 3}}

    from platform.data.synthetic import EngineTeacher
    teacher = EngineTeacher(engine=_FakeEngine(), tokenizer=_FakeTokenizer(), name="fake",
                            request_kwargs={"max_new_tokens": 3, "temperature": 0.0})
    response = teacher.generate("hi")
    assert response.startswith("ok ")

    pol = TextbookPolicy()
    fac = SyntheticFactory(teacher=teacher, policy=pol, seed=0)
    out = list(fac.generate(2))
    assert len(out) == 2
    assert all(r.accepted for r in out)
    assert all(r.response.startswith("ok ") for r in out)

"""Required real TRL/Transformers integration; all artifacts are built locally."""
from __future__ import annotations

from copy import deepcopy
import math

from datasets import Dataset
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from train import build_sft_config, build_trainer, validate_supervision


TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}assistant "
    "{% generation %}{{ message['content'] }} [EOS] {% endgeneration %}"
    "{% else %}{{ message['role'] }} {{ message['content'] }} [EOS] {% endif %}"
    "{% endfor %}"
)


def local_fixture(tmp_path, scope):
    vocabulary = {text: i for i, text in enumerate(
        ["[UNK]", "[PAD]", "[EOS]", "user", "assistant", "question", "red",
         "blue", "one", "two", "yes", "no"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       pad_token="[PAD]", eos_token="[EOS]",
                                       chat_template=TEMPLATE, model_max_length=64)
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=64, n_ctx=64, n_embd=16,
        n_layer=1, n_head=2, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None, use_cache=False,
    ))

    def row(question, answer):
        return [{"role": "user", "content": "question " + question},
                {"role": "assistant", "content": answer}]

    train = Dataset.from_dict({"messages": [row("red", "yes"), row("blue", "no")],
                               "source_id": ["train-red", "train-blue"]})
    validation = Dataset.from_dict({"messages": [row("one", "yes")],
                                    "source_id": ["validation-one"]})
    config = {
        "out_dir": str(tmp_path / scope), "seed": 7, "method": "full",
        "model": {"dtype": "float32", "attn_impl": "sdpa"},
        "train": {"loss_scope": scope, "epochs": 1., "max_steps": 1,
                  "batch_size": 2, "grad_accum": 1, "lr": 0.001,
                  "warmup_ratio": 0., "weight_decay": 0., "grad_clip": 1.,
                  "log_every": 1, "save_every": 100, "eval_every": 1,
                  "max_seq_len": 32, "packing": False, "use_cpu": True,
                  "gradient_checkpointing": False},
    }
    return model, tokenizer, train, validation, config


@pytest.mark.parametrize("scope", ["all_tokens", "assistant_only"])
def test_actual_sft_step_masks_validation_and_parameter_update(tmp_path, scope):
    model, tokenizer, data, validation, cfg = local_fixture(tmp_path, scope)
    trainer = build_trainer(model, tokenizer, data, cfg, eval_ds=validation)
    batch = next(iter(trainer.get_train_dataloader()))
    assert (batch["labels"] != -100).any()
    if scope == "assistant_only":
        user_id = tokenizer.convert_tokens_to_ids("user")
        assert (batch["labels"][batch["input_ids"] == user_id] == -100).all()
        assert (batch["labels"] == -100).any()
    assert (batch["labels"][batch["input_ids"] == tokenizer.pad_token_id] == -100).all()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = trainer.train()
    assert result.global_step == 1
    assert math.isfinite(result.metrics["train_loss"])
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert math.isfinite(trainer.evaluate()["eval_loss"])
    assert len(trainer.eval_dataset) == 1 and len(trainer.train_dataset) == 2


def test_assistant_mask_requires_supported_template_and_surviving_targets(tmp_path):
    _, tok, data, validation, cfg = local_fixture(tmp_path, "assistant_only")
    original = tok.chat_template
    tok.chat_template = "{{ messages }}"
    with pytest.raises(ValueError, match="generation"):
        validate_supervision(tok, (data, validation), cfg)
    tok.chat_template = original
    cfg["train"]["max_seq_len"] = 2
    with pytest.raises(ValueError, match="assistant targets"):
        validate_supervision(tok, (data, validation), cfg)


def test_supervision_and_packing_never_silently_change_semantics(tmp_path):
    _, tok, data, validation, cfg = local_fixture(tmp_path, "all_tokens")
    with pytest.raises(ValueError, match="validation"):
        validate_supervision(tok, (data, None), cfg)
    invalid = deepcopy(cfg)
    invalid["train"]["loss_scope"] = "implicit"
    with pytest.raises(ValueError, match="loss_scope"):
        build_sft_config(invalid)
    invalid = deepcopy(cfg)
    invalid["train"]["packing"] = True
    with pytest.raises(ValueError, match="boundary-aware"):
        build_sft_config(invalid)


def test_actual_local_dpo_step_updates_parameters(tmp_path):
    from cf_pref.dpo_train import build_pref_trainer

    model, tok, _, _, cfg = local_fixture(tmp_path, "all_tokens")
    cfg["out_dir"] = str(tmp_path / "dpo")
    cfg["pref"] = {"objective": "dpo", "loss_type": "sigmoid", "beta": 0.1}
    pairs = Dataset.from_dict({
        "prompt": ["user question red [EOS] assistant ", "user question blue [EOS] assistant "],
        "chosen": ["yes [EOS]", "no [EOS]"],
        "rejected": ["no [EOS]", "yes [EOS]"],
    })
    trainer = build_pref_trainer(model, tok, pairs, cfg)
    assert trainer.ref_model is not model
    assert all(not parameter.requires_grad for parameter in trainer.ref_model.parameters())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter, reference in zip(model.parameters(), trainer.ref_model.parameters()):
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        assert parameter.data_ptr() != reference.data_ptr()
    result = trainer.train()
    assert result.global_step == 1 and math.isfinite(result.metrics["train_loss"])
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_actual_local_grpo_step_with_nonexecuting_reward_double(tmp_path):
    from cf_rl.grpo_train import build_grpo_trainer

    model, tok, _, _, cfg = local_fixture(tmp_path, "all_tokens")
    cfg["out_dir"] = str(tmp_path / "grpo")
    cfg["grpo"] = {"num_generations": 2, "max_completion_length": 3,
                   "temperature": 1.0, "beta": 0.0, "num_iterations": 1,
                   "loss_type": "grpo", "use_vllm": False}
    prompts = Dataset.from_dict({"prompt": ["question red", "question blue"]})
    calls = []

    def nonexecuting_fixture_reward(completions, **kwargs):
        # Exercise only the actual generation/optimizer API, not code execution,
        # reward quality, a Python sandbox, or an external model's capabilities.
        calls.append(len(completions))
        return [float(index % 2) for index in range(len(completions))]

    trainer = build_grpo_trainer(model, tok, prompts, [nonexecuting_fixture_reward], cfg)
    result = trainer.train()
    assert result.global_step == 1 and math.isfinite(result.metrics["train_loss"])
    assert calls and all(count == 2 for count in calls)
    assert trainer.optimizer.state

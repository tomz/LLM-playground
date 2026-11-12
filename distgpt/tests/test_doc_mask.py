"""Document masking: packed-sequence logits equal isolated-document logits."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distgpt.model.config import ModelConfig
from distgpt.model.transformer import GPT, document_layout

EOT = 0


def _model(**kw):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, max_seq_len=32, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=32, d_ffn=64, doc_mask_eot=EOT, **kw)
    return GPT(cfg)


def test_layout():
    doc, pos = document_layout(torch.tensor([[5, EOT, 7, 8, EOT, 9]]), EOT)
    assert doc.tolist() == [[0, 0, 1, 1, 1, 2]] and pos.tolist() == [[0, 1, 0, 1, 2, 0]]


@pytest.mark.parametrize("qk_norm", [False, True])
def test_packed_matches_isolated(qk_norm):
    m = _model(qk_norm=qk_norm).eval()
    docs = [torch.tensor([3, 4, 5, EOT]), torch.tensor([9, EOT]), torch.tensor([11, 12, 13])]
    packed = torch.cat(docs)[None]
    with torch.no_grad():
        full = m.lm_head(m.final_norm(_hidden(m, packed)))
        s = 0
        for d in docs:
            alone = m.lm_head(m.final_norm(_hidden(m, d[None])))
            torch.testing.assert_close(full[0, s:s + len(d)], alone[0], rtol=1e-5, atol=1e-5)
            s += len(d)


def _hidden(m, idx):
    cap = {}
    h = m.final_norm.register_forward_hook(lambda _m, i, _o: cap.setdefault("x", i[0]))
    m(idx, targets=idx)
    h.remove()
    return cap["x"]


def test_training_with_activation_checkpointing_has_finite_grads():
    m = _model().train()
    m.activation_ckpt = "full"
    x = torch.randint(1, 64, (2, 16)); x[:, 6] = EOT
    _, loss = m(x, targets=x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)


def test_mla_rejected():
    m = _model(attn_kind="mla", mla_kv_latent_dim=16)
    with pytest.raises(NotImplementedError):
        m(torch.tensor([[1, 2, EOT, 3]]), targets=torch.tensor([[1, 2, EOT, 3]]))

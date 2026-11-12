"""Document masking: packed-sequence logits must equal per-document logits."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig, document_layout

EOT = 0


def _model(**kw):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32,
                    d_ffn=64, doc_mask_eot=EOT, **kw)
    return GPT(cfg).eval()


def test_layout_eot_closes_its_document():
    idx = torch.tensor([[5, 6, EOT, 7, EOT, 8, 9]])
    doc, pos = document_layout(idx, EOT)
    assert doc.tolist() == [[0, 0, 0, 1, 1, 2, 2]]
    assert pos.tolist() == [[0, 1, 2, 0, 1, 0, 1]]


@pytest.mark.parametrize("arch", [dict(), dict(pos_kind="rope", norm_kind="rmsnorm", mlp_kind="swiglu", qk_norm=True)])
def test_packed_logits_match_isolated_documents(arch):
    m = _model(**arch)
    docs = [torch.tensor([3, 4, 5, EOT]), torch.tensor([9, 10, EOT]), torch.tensor([11, 12, 13, 14, 15])]
    packed = torch.cat(docs)[None]
    with torch.no_grad():
        full, _ = m(packed, return_full_logits=True)
        start = 0
        for d in docs:
            alone, _ = m(d[None], return_full_logits=True)
            torch.testing.assert_close(full[0, start:start + len(d)], alone[0], rtol=1e-5, atol=1e-5)
            start += len(d)


def test_without_flag_documents_leak_and_training_still_works():
    m = _model()
    m.cfg.doc_mask_eot = None
    packed = torch.tensor([[3, 4, EOT, 9, 10]])
    with torch.no_grad():
        leaky, _ = m(packed, return_full_logits=True)
        alone, _ = m(torch.tensor([[9, 10]]), return_full_logits=True)
    assert not torch.allclose(leaky[0, 3:], alone[0], atol=1e-4)
    m = _model(pos_kind="rope").train()
    x = torch.randint(1, 64, (2, 16)); x[:, 7] = EOT
    _, loss = m(x, targets=x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

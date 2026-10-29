from __future__ import annotations

import torch
import torch.nn.functional as F

from areno.engine.config import ModelConfig
from areno.models.qwen3.model import Qwen3ForCausalLM


def _model(*, tied: bool) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        ModelConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            tie_word_embeddings=tied,
            dtype=torch.float32,
        )
    )


def test_tied_qwen3_updates_embedding_and_output_as_one_parameter():
    model = _model(tied=True)
    assert model.lm_head.weight is model.embed_tokens.weight
    embedding = model.embed_tokens.weight
    original = embedding.detach().clone()
    loss = F.linear(F.embedding(torch.tensor([1, 2]), embedding), model.lm_head.weight).sum()
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(original, embedding)
    torch.testing.assert_close(model.lm_head.weight, embedding, atol=0, rtol=0)


def test_untied_qwen3_keeps_separate_parameters():
    model = _model(tied=False)
    assert model.lm_head.weight is not model.embed_tokens.weight

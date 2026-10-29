from __future__ import annotations

from dataclasses import replace
from inspect import Parameter, signature
from types import SimpleNamespace

import pytest
import torch

import areno.engine.api as engine_api
import areno.engine.inference as inference_mod
from areno.engine.api import _shared_prompt_kv_plan
from areno.engine.config import RuntimeConfig
from areno.engine.data import SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.inference import InferenceManager
from areno.engine.layers.attention_backend.infer import _store_prefill_cache
from areno.engine.protocol import RolloutPayload
from areno.engine.runtime.metadata import InferMeta
from areno.engine.worker import _rollout_payloads_compatible
from areno.models.qwen3.model import Qwen3ForCausalLM
from tests.helpers import PatchedContext
from tests.test_inference_scheduler_cpu import _FakeInferenceManager


def _state(prompt, samples=4):
    return InferenceBatchState(
        prompts=[list(prompt) for _ in range(samples)],
        max_new_tokens=4,
        max_running_seqs=samples,
        max_cache_len=len(prompt) + 4,
        max_prefill_tokens=100,
        kv_block_size=4,
        num_cache_blocks=16,
        prompt_group_size=samples,
    )


def test_shared_prompt_config_does_not_change_existing_positional_arguments():
    parameters = signature(RuntimeConfig).parameters
    assert parameters["decode_graph_buckets"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["share_prompt_kv"].kind is Parameter.KEYWORD_ONLY


def test_shared_prompt_prefills_once_and_copies_partial_tail_on_write():
    state = _state([1, 2, 3, 4, 5, 6])
    payload = state.build_prefill_payload()

    assert payload["input_ids"].tolist() == [1, 2, 3, 4, 5, 6]
    assert payload["sample_indices"].tolist() == [5, 5, 5, 5]
    assert state._last_active_ids == [0, 1, 2, 3]
    original = list(state._seq_to_blocks[0])
    assert all(state._seq_to_blocks[row] == original for row in range(4))
    assert [state._block_refcounts[block] for block in original] == [4, 4]

    copies = state.ensure_decode_blocks([0, 1, 2, 3], [6, 6, 6, 6])
    assert len(copies) == 3
    assert all(source == original[1] and count == 2 for source, _, count in copies)
    assert len({state._seq_to_blocks[row][1] for row in range(4)}) == 4
    assert all(state._seq_to_blocks[row][0] == original[0] for row in range(4))


def test_shared_blocks_remain_owned_until_last_cancelled_or_finished_row():
    state = _state([1, 2, 3, 4])
    state.build_prefill_payload()
    shared_block = state._seq_to_blocks[0][0]
    state.ensure_decode_blocks([0, 1, 2, 3], [4, 4, 4, 4])
    manager = InferenceManager(SimpleNamespace(device=torch.device("cpu")))

    manager._free_rollout_rows(state, torch.tensor([1, 3]))
    assert shared_block not in state._free_blocks
    assert state._block_refcounts[shared_block] == 2
    manager._free_rollout_rows(state, torch.tensor([0]))
    assert state._block_refcounts[shared_block] == 1
    manager._free_rollout_rows(state, torch.tensor([2]))
    assert shared_block not in state._block_refcounts
    assert len(state._free_blocks) == state.num_cache_blocks
    assert len(set(state._free_blocks)) == state.num_cache_blocks
    assert len(state._free_recurrent_slots) == state.max_running_seqs


def test_chunked_prefill_branches_only_after_the_final_chunk():
    state = _state([1, 2, 3, 4, 5])
    state.max_prefill_tokens = 2
    first = state.build_prefill_payload()
    assert first["sample_indices"].numel() == 0
    assert state._last_active_ids == []
    second = state.build_prefill_payload()
    assert second["sample_indices"].numel() == 0
    final = state.build_prefill_payload()
    assert final["sample_indices"].tolist() == [0, 0, 0, 0]
    assert state._last_active_ids == [0, 1, 2, 3]


def test_distinct_prompt_groups_do_not_share_blocks_or_logits():
    state = InferenceBatchState(
        prompts=[[1, 2], [1, 2], [3, 4, 5], [3, 4, 5]],
        max_new_tokens=2,
        max_running_seqs=4,
        kv_block_size=4,
        num_cache_blocks=8,
        prompt_group_size=2,
    )
    payload = state.build_prefill_payload()
    assert payload["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert payload["sample_indices"].tolist() == [1, 1, 4, 4]
    assert payload["block_table"].shape[0] == 2
    assert state._seq_to_blocks[0] == state._seq_to_blocks[1]
    assert state._seq_to_blocks[2] == state._seq_to_blocks[3]
    assert state._seq_to_blocks[0] != state._seq_to_blocks[2]
    expanded = state.expand_shared_prefill(payload)
    assert expanded["input_ids"].tolist() == [1, 2, 1, 2, 3, 4, 5, 3, 4, 5]
    assert expanded["cu_seqlens"].tolist() == [0, 2, 4, 7, 10]
    assert expanded["sample_indices"].tolist() == [1, 3, 6, 9]
    assert expanded["cache_write_indices"].tolist() == [0, 1, 4, 5, 6]
    assert expanded["prefill_seq_ids"] == [0, 1, 2, 3]


def test_prefill_cache_writes_only_the_shared_prompt_rows():
    keys = torch.arange(16).reshape(8, 1, 2).float()
    values = keys + 100
    k_cache = torch.full((2, 4, 1, 2), -1.0)
    v_cache = k_cache.clone()
    meta = InferMeta(
        mode="prefill",
        cache_block_ids=torch.tensor([0, 0, 1, 1]),
        cache_block_offsets=torch.tensor([0, 1, 0, 1]),
        cache_write_indices=torch.tensor([0, 1, 4, 5]),
    )
    _store_prefill_cache(keys, values, k_cache, v_cache, meta)
    assert torch.equal(k_cache[:, :2, 0], keys[[0, 1, 4, 5], 0].reshape(2, 2, 2))
    assert torch.equal(v_cache[:, :2, 0], values[[0, 1, 4, 5], 0].reshape(2, 2, 2))
    assert torch.all(k_cache[:, 2:] == -1)


def test_tail_copy_copies_only_the_populated_prefix():
    k = torch.arange(24).reshape(3, 4, 2)
    v = k.clone() + 100
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(k_cache=k, v_cache=v))])
    old_k, old_v = k[1].clone(), v[1].clone()

    Qwen3ForCausalLM.copy_kv_cache_blocks(model, [(0, 1, 2)])

    assert torch.equal(k[1, :2], k[0, :2])
    assert torch.equal(v[1, :2], v[0, :2])
    assert torch.equal(k[1, 2:], old_k[2:])
    assert torch.equal(v[1, 2:], old_v[2:])


def _config(*, model_type="qwen3", tp_size=1, dp_size=1):
    return SimpleNamespace(
        runtime=SimpleNamespace(share_prompt_kv=True, device_type="cuda", kv_block_size=4),
        model=SimpleNamespace(model_type=model_type),
        tp_size=tp_size,
        dp_size=dp_size,
    )


def test_pool_plan_reserves_shared_full_blocks_and_private_tails():
    prompts = [[1, 2, 3, 4, 5, 6]] * 4
    assert _shared_prompt_kv_plan(_config(), prompts, None, 4, 4, 4, 12, 24) == (4, 9)
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8]] * 4
    assert _shared_prompt_kv_plan(_config(), prompts, None, 4, 4, 4, 12, 32) == (4, 6)


def test_shared_batch_is_never_refilled_with_another_request():
    payload = RolloutPayload(
        prompts_by_dp=[[[1]]],
        prompt_indices_by_dp=[[0]],
        prompt_features_by_dp=None,
        max_new_tokens=4,
        eos_token_id=None,
        sampling_params=SamplingParams(),
        max_running_seqs=4,
        max_cache_len=8,
        max_blocks_per_seq=2,
        max_prefill_tokens=8,
        num_blocks=8,
        block_size=4,
    )
    shared = replace(payload, prompt_group_size=4)
    assert _rollout_payloads_compatible(payload, payload)
    assert not _rollout_payloads_compatible(shared, payload)
    assert not _rollout_payloads_compatible(payload, shared)


@pytest.mark.parametrize(
    ("config", "prompts", "features", "capacity", "prefill_budget"),
    [
        (_config(model_type="qwen3_5"), [[1, 2]] * 4, None, 4, 8),
        (_config(tp_size=2), [[1, 2]] * 4, None, 4, 8),
        (_config(), [[1, 2]] * 4, [{"mrope_position_ids": [0, 1]}] * 4, 4, 8),
        (_config(), [[1, 2]] * 4, None, 2, 8),
        (_config(), [[1, 2], [1, 3], [1, 2], [1, 2]], None, 4, 8),
        (_config(), [[1, 2]] * 4, None, 4, 7),
    ],
)
def test_unsupported_batches_fall_back_to_independent_caches(
    config, prompts, features, capacity, prefill_budget, monkeypatch
):
    warnings = []
    monkeypatch.setattr(engine_api.logger, "warning", lambda message, reason: warnings.append(message % reason))
    assert _shared_prompt_kv_plan(config, prompts, features, 4, capacity, 4, 8, prefill_budget) == (1, 8)
    assert "using independent caches" in warnings[0]


def test_shared_rollout_keeps_samples_separate_and_releases_after_cancellation():
    class Manager(_FakeInferenceManager):
        def _infer_next_token_tensor(self, payload):
            return torch.arange(1, 5), torch.arange(4, dtype=torch.float32)

    manager = Manager()
    copies = []
    manager.model = SimpleNamespace(copy_kv_cache_blocks=lambda ops: copies.extend(ops))
    state = _state([1, 2, 3, 4, 5, 6])
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            cancel_flags=torch.tensor([False, True, False, False]),
            cancel_indices=[0, 1, 2, 3],
        )

    assert state.generated == [[1, 2, 3, 4], [0], [3, 4, 5, 6], [4, 5, 6, 7]]
    assert len(copies) == 2
    assert not state._block_refcounts
    assert len(state._free_blocks) == state.num_cache_blocks

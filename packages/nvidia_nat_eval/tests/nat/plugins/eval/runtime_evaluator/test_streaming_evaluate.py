# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for streaming evaluators (TTFT, TPS, ITL)."""

import pytest

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.token_usage import TokenUsageBaseModel
from nat.plugins.eval.runtime_evaluator.streaming_evaluate import AverageITLEvaluator
from nat.plugins.eval.runtime_evaluator.streaming_evaluate import AverageTPSEvaluator
from nat.plugins.eval.runtime_evaluator.streaming_evaluate import AverageTTFTEvaluator


def make_intermediate_step(
    event_type: IntermediateStepType,
    timestamp: float,
    uuid: str,
    *,
    completion_tokens: int = 0,
    prompt_tokens: int = 0,
    total_tokens: int = 0,
) -> IntermediateStep:
    """Factory to build IntermediateStep objects for streaming evaluator tests.

    Args:
        event_type: LLM_START, LLM_NEW_TOKEN, or LLM_END.
        timestamp: event_timestamp for the step.
        uuid: UUID for grouping events into LLM calls.
        completion_tokens: For LLM_END events; used for TPS/ITL.
        prompt_tokens: For LLM_END events; used with total_tokens as fallback.
        total_tokens: For LLM_END events when completion_tokens is 0.

    Returns:
        IntermediateStep with the given parameters.
    """
    usage_info = None
    if event_type == IntermediateStepType.LLM_END:
        usage_info = UsageInfo(
            token_usage=TokenUsageBaseModel(
                prompt_tokens=prompt_tokens or 10,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            ),
        )

    payload = IntermediateStepPayload(
        event_type=event_type,
        event_timestamp=timestamp,
        UUID=uuid,
        usage_info=usage_info,
    )
    return IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(function_id="test-fn", function_name="test_fn"),
        payload=payload,
    )


def make_eval_item(
    item_id: str = "test_1",
    trajectory: list[IntermediateStep] | None = None,
) -> EvalInputItem:
    """Create an EvalInputItem with minimal required fields and optional trajectory."""
    return EvalInputItem(
        id=item_id,
        input_obj="test input",
        expected_output_obj="test expected",
        output_obj=None,
        expected_trajectory=[],
        trajectory=trajectory or [],
        full_dataset_entry={},
    )


# ============== AverageTTFTEvaluator ==============


@pytest.mark.asyncio
async def test_ttft_from_new_token_events():
    """Single LLM call with START, 3 NEW_TOKENs, END. TTFT = first_token_ts - start_ts."""
    start_ts = 100.0
    first_token_ts = 100.5
    uuid_val = "call-1"
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, start_ts, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, first_token_ts, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 100.6, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 100.7, uuid_val),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            101.0,
            uuid_val,
            completion_tokens=3,
            prompt_tokens=10,
        ),
    ]
    item = make_eval_item("ttft_1", steps)
    evaluator = AverageTTFTEvaluator()
    result = await evaluator.evaluate_item(item)
    expected_ttft = first_token_ts - start_ts  # 0.5
    assert result.score == pytest.approx(expected_ttft, abs=1e-5)
    assert result.reasoning["num_llm_calls"] == 1
    assert result.reasoning["has_streaming_tokens"] is True


@pytest.mark.asyncio
async def test_ttft_multiple_calls():
    """Two LLM calls with different UUIDs, returns average TTFT."""
    # Call 1: TTFT = 0.2
    # Call 2: TTFT = 0.4
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, 200.0, "call-1"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 200.2, "call-1"),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            201.0,
            "call-1",
            completion_tokens=1,
            prompt_tokens=5,
        ),
        make_intermediate_step(IntermediateStepType.LLM_START, 300.0, "call-2"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 300.4, "call-2"),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            301.0,
            "call-2",
            completion_tokens=1,
            prompt_tokens=5,
        ),
    ]
    item = make_eval_item("ttft_multi", steps)
    evaluator = AverageTTFTEvaluator()
    result = await evaluator.evaluate_item(item)
    expected_avg = (0.2 + 0.4) / 2
    assert result.score == pytest.approx(expected_avg, abs=1e-5)
    assert result.reasoning["num_llm_calls"] == 2


@pytest.mark.asyncio
async def test_ttft_fallback_no_new_tokens():
    """START + END only, no NEW_TOKEN events - falls back to duration."""
    start_ts = 50.0
    end_ts = 52.0
    uuid_val = "call-fallback"
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, start_ts, uuid_val),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            end_ts,
            uuid_val,
            completion_tokens=10,
            prompt_tokens=20,
        ),
    ]
    item = make_eval_item("ttft_fallback", steps)
    evaluator = AverageTTFTEvaluator()
    result = await evaluator.evaluate_item(item)
    # Fallback uses full duration
    assert result.score == pytest.approx(end_ts - start_ts, abs=1e-5)
    assert result.reasoning["has_streaming_tokens"] is False


@pytest.mark.asyncio
async def test_ttft_empty_trajectory():
    """Empty trajectory yields score 0.0."""
    item = make_eval_item("ttft_empty", [])
    evaluator = AverageTTFTEvaluator()
    result = await evaluator.evaluate_item(item)
    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0


# ============== AverageTPSEvaluator ==============


@pytest.mark.asyncio
async def test_tps_basic():
    """100 completion tokens in 2 seconds = 50 TPS."""
    start_ts = 10.0
    end_ts = 12.0
    uuid_val = "tps-1"
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, start_ts, uuid_val),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            end_ts,
            uuid_val,
            completion_tokens=100,
            prompt_tokens=10,
        ),
    ]
    item = make_eval_item("tps_basic", steps)
    evaluator = AverageTPSEvaluator()
    result = await evaluator.evaluate_item(item)
    expected_tps = 100 / 2
    assert result.score == pytest.approx(expected_tps, abs=0.01)
    assert result.reasoning["num_llm_calls"] == 1


@pytest.mark.asyncio
async def test_tps_multiple_calls():
    """Two calls, returns average TPS."""
    # Call 1: 50 tokens in 1s = 50 TPS
    # Call 2: 100 tokens in 2s = 50 TPS
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, 0.0, "tps-a"),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            1.0,
            "tps-a",
            completion_tokens=50,
            prompt_tokens=10,
        ),
        make_intermediate_step(IntermediateStepType.LLM_START, 5.0, "tps-b"),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            7.0,
            "tps-b",
            completion_tokens=100,
            prompt_tokens=10,
        ),
    ]
    item = make_eval_item("tps_multi", steps)
    evaluator = AverageTPSEvaluator()
    result = await evaluator.evaluate_item(item)
    expected_avg = (50 + 50) / 2
    assert result.score == pytest.approx(expected_avg, abs=0.01)
    assert result.reasoning["num_llm_calls"] == 2


@pytest.mark.asyncio
async def test_tps_zero_tokens():
    """Zero completion tokens yields 0.0 score (no valid TPS, avg over empty = 0)."""
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, 0.0, "zero"),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            1.0,
            "zero",
            completion_tokens=0,
            prompt_tokens=10,
            total_tokens=10,
        ),
    ]
    item = make_eval_item("tps_zero", steps)
    evaluator = AverageTPSEvaluator()
    result = await evaluator.evaluate_item(item)
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_tps_empty_trajectory():
    """Empty trajectory yields score 0.0."""
    item = make_eval_item("tps_empty", [])
    evaluator = AverageTPSEvaluator()
    result = await evaluator.evaluate_item(item)
    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0


# ============== AverageITLEvaluator ==============


@pytest.mark.asyncio
async def test_itl_from_new_token_events():
    """4 NEW_TOKEN events at 0.1s intervals = 0.1s ITL."""
    base_ts = 500.0
    uuid_val = "itl-1"
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, base_ts, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, base_ts + 0.0, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, base_ts + 0.1, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, base_ts + 0.2, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, base_ts + 0.3, uuid_val),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            base_ts + 0.5,
            uuid_val,
            completion_tokens=4,
            prompt_tokens=10,
        ),
    ]
    item = make_eval_item("itl_basic", steps)
    evaluator = AverageITLEvaluator()
    result = await evaluator.evaluate_item(item)
    # Gaps: 0.1, 0.1, 0.1 -> mean = 0.1
    assert result.score == pytest.approx(0.1, abs=1e-5)
    assert result.reasoning["has_streaming_tokens"] is True


@pytest.mark.asyncio
async def test_itl_multiple_calls():
    """Two LLM calls, returns average ITL."""
    # Call 1: gaps 0.1, 0.1 -> mean 0.1
    # Call 2: gaps 0.2, 0.2 -> mean 0.2
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, 0.0, "itl-a"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 0.0, "itl-a"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 0.1, "itl-a"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 0.2, "itl-a"),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            0.5,
            "itl-a",
            completion_tokens=3,
            prompt_tokens=5,
        ),
        make_intermediate_step(IntermediateStepType.LLM_START, 10.0, "itl-b"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 10.0, "itl-b"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 10.2, "itl-b"),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 10.4, "itl-b"),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            11.0,
            "itl-b",
            completion_tokens=3,
            prompt_tokens=5,
        ),
    ]
    item = make_eval_item("itl_multi", steps)
    evaluator = AverageITLEvaluator()
    result = await evaluator.evaluate_item(item)
    expected_avg = (0.1 + 0.2) / 2
    assert result.score == pytest.approx(expected_avg, abs=1e-5)
    assert result.reasoning["num_llm_calls"] == 2


@pytest.mark.asyncio
async def test_itl_fallback_no_new_tokens():
    """Uses duration / completion_tokens when no NEW_TOKEN events."""
    start_ts = 0.0
    end_ts = 2.0
    completion_tokens = 100
    uuid_val = "itl-fallback"
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, start_ts, uuid_val),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            end_ts,
            uuid_val,
            completion_tokens=completion_tokens,
            prompt_tokens=10,
        ),
    ]
    item = make_eval_item("itl_fallback", steps)
    evaluator = AverageITLEvaluator()
    result = await evaluator.evaluate_item(item)
    expected_itl = (end_ts - start_ts) / completion_tokens  # 0.02
    assert result.score == pytest.approx(expected_itl, abs=1e-5)
    assert result.reasoning["has_streaming_tokens"] is False


@pytest.mark.asyncio
async def test_itl_single_new_token():
    """Only 1 NEW_TOKEN - cannot compute ITL from gaps, falls back to duration/completion_tokens or yields 0."""
    # With completion_tokens=1, fallback returns None (completion_tokens > 1 required).
    # So this call contributes nothing to itls; avg_itl = 0.0 for single empty-contributor case.
    start_ts = 0.0
    end_ts = 1.0
    uuid_val = "itl-single"
    steps = [
        make_intermediate_step(IntermediateStepType.LLM_START, start_ts, uuid_val),
        make_intermediate_step(IntermediateStepType.LLM_NEW_TOKEN, 0.5, uuid_val),
        make_intermediate_step(
            IntermediateStepType.LLM_END,
            end_ts,
            uuid_val,
            completion_tokens=1,
            prompt_tokens=5,
        ),
    ]
    item = make_eval_item("itl_single", steps)
    evaluator = AverageITLEvaluator()
    result = await evaluator.evaluate_item(item)
    # Single token: no gaps, fallback requires completion_tokens > 1, so itl is None.
    # itls is empty -> avg_itl = 0.0
    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0


@pytest.mark.asyncio
async def test_itl_empty_trajectory():
    """Empty trajectory yields score 0.0."""
    item = make_eval_item("itl_empty", [])
    evaluator = AverageITLEvaluator()
    result = await evaluator.evaluate_item(item)
    assert result.score == 0.0
    assert result.reasoning["num_llm_calls"] == 0

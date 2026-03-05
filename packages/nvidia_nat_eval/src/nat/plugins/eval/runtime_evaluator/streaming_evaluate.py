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

"""
Streaming-aware runtime evaluators for TTFT, ITL, and TPS.

These evaluators derive per-request metrics from IntermediateStep event
sequences (LLM_START, LLM_NEW_TOKEN, LLM_END) grouped by UUID.

When LLM_NEW_TOKEN events are available (LangChain streaming), TTFT and ITL
are computed directly from token timestamps.  When they are not available,
fallback heuristics use LLM_START/LLM_END timing and completion token counts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator
from nat.plugins.eval.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor


@dataclass
class _StreamingCallData:
    """Accumulated timing data for a single LLM call (grouped by UUID)."""

    start_ts: float | None = None
    end_ts: float | None = None
    token_timestamps: list[float] = field(default_factory=list)
    completion_tokens: int = 0

    @property
    def duration(self) -> float | None:
        if self.start_ts is None or self.end_ts is None:
            return None
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def ttft(self) -> float | None:
        """Time to first token: first NEW_TOKEN timestamp minus START timestamp."""
        if self.start_ts is None:
            return None
        if self.token_timestamps:
            return max(0.0, min(self.token_timestamps) - self.start_ts)
        # Fallback: use span_event_timestamp (LLM_END carries start time)
        # or approximate as total duration (pessimistic)
        return self.duration

    @property
    def itl(self) -> float | None:
        """Mean inter-token latency from consecutive NEW_TOKEN timestamps."""
        if len(self.token_timestamps) >= 2:
            sorted_ts = sorted(self.token_timestamps)
            gaps = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]
            return sum(gaps) / len(gaps) if gaps else None
        # Fallback: estimate from duration and token count
        dur = self.duration
        if dur is not None and self.completion_tokens > 1:
            return dur / self.completion_tokens
        return None

    @property
    def tps(self) -> float | None:
        """Tokens per second: completion_tokens / duration."""
        dur = self.duration
        if dur is not None and dur > 0 and self.completion_tokens > 0:
            return self.completion_tokens / dur
        return None


def _collect_call_data(item: EvalInputItem) -> dict[str, _StreamingCallData]:
    """Group trajectory steps by UUID into per-call streaming data."""
    calls: dict[str, _StreamingCallData] = defaultdict(_StreamingCallData)

    for step in (IntermediatePropertyAdaptor.from_intermediate_step(s) for s in item.trajectory):
        if step.event_type == IntermediateStepType.LLM_START:
            calls[step.UUID].start_ts = step.event_timestamp
        elif step.event_type == IntermediateStepType.LLM_NEW_TOKEN:
            calls[step.UUID].token_timestamps.append(step.event_timestamp)
        elif step.event_type == IntermediateStepType.LLM_END:
            calls[step.UUID].end_ts = step.event_timestamp
            tokens = step.token_usage.completion_tokens
            if tokens == 0:
                tokens = step.token_usage.total_tokens - step.token_usage.prompt_tokens
            calls[step.UUID].completion_tokens = max(0, tokens)

    return calls


class AverageTTFTEvaluator(BaseEvaluator):
    """
    Average Time to First Token across all LLM calls in the item.

    Score is the mean TTFT in seconds (lower is better).
    Uses LLM_NEW_TOKEN timestamps when available; falls back to full
    LLM_START-to-LLM_END latency when streaming events are absent.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Avg TTFT")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        calls = _collect_call_data(item)

        ttfts = [c.ttft for c in calls.values() if c.ttft is not None]
        has_streaming = any(c.token_timestamps for c in calls.values())
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

        reasoning = {
            "num_llm_calls": len(ttfts),
            "has_streaming_tokens": has_streaming,
            "ttfts": [round(t, 6) for t in ttfts],
        }
        return EvalOutputItem(id=item.id, score=round(avg_ttft, 6), reasoning=reasoning)


class AverageTPSEvaluator(BaseEvaluator):
    """
    Average Tokens Per Second across all LLM calls in the item.

    Score is the mean TPS (higher is better).
    Computed as completion_tokens / (LLM_END - LLM_START) per call.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Avg TPS")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        calls = _collect_call_data(item)

        tps_values = [c.tps for c in calls.values() if c.tps is not None]
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0

        reasoning = {
            "num_llm_calls": len(tps_values),
            "tps_values": [round(t, 2) for t in tps_values],
        }
        return EvalOutputItem(id=item.id, score=round(avg_tps, 2), reasoning=reasoning)


class AverageITLEvaluator(BaseEvaluator):
    """
    Average Inter-Token Latency across all LLM calls in the item.

    Score is the mean ITL in seconds (lower is better).
    Uses consecutive LLM_NEW_TOKEN timestamp deltas when available;
    falls back to duration / completion_tokens when streaming events are absent.
    """

    def __init__(self, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating Avg ITL")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        calls = _collect_call_data(item)

        itls = [c.itl for c in calls.values() if c.itl is not None]
        has_streaming = any(c.token_timestamps for c in calls.values())
        avg_itl = sum(itls) / len(itls) if itls else 0.0

        reasoning = {
            "num_llm_calls": len(itls),
            "has_streaming_tokens": has_streaming,
            "itls": [round(t, 6) for t in itls],
        }
        return EvalOutputItem(id=item.id, score=round(avg_itl, 6), reasoning=reasoning)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Replay logger for routing decisions and feedback.

Writes JSONL files (one JSON object per line) that enable post-hoc replay
of routing decisions and learner state evolution. Controlled by the
REPLAY_LOG_DIR environment variable — when unset, no logger is created
and the router paths have zero overhead.

Output files:
    routing_decisions.jsonl  — full scoring state for every routing decision
    routing_feedback.jsonl   — reward signal and learner state after update
    run_metadata.json        — run-level metadata written once at startup
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ReplayLogger:
    """Thread-safe JSONL writer for routing replay logs.

    Each ``log_decision`` / ``log_feedback`` call serializes one JSON object,
    appends it as a line to the corresponding file, and flushes immediately.
    The low per-request write volume makes per-line flush acceptable.
    """

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        router_type: str,
        config: dict[str, Any],
    ) -> None:
        self.run_id = run_id
        self._lock = threading.Lock()

        os.makedirs(output_dir, exist_ok=True)

        decisions_path = os.path.join(output_dir, "routing_decisions.jsonl")
        feedback_path = os.path.join(output_dir, "routing_feedback.jsonl")

        self._decisions_fh = open(decisions_path, "a", encoding="utf-8")
        self._feedback_fh = open(feedback_path, "a", encoding="utf-8")

        metadata = {
            "run_id": run_id,
            "router_type": router_type,
            "dataset": None,
            "dataset_hash": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
        }
        metadata_path = os.path.join(output_dir, "run_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "ReplayLogger active: run_id=%s  dir=%s", run_id, output_dir,
        )

    def log_decision(self, record: dict[str, Any]) -> None:
        """Append a routing decision record to ``routing_decisions.jsonl``."""
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._decisions_fh.write(line)
            self._decisions_fh.write("\n")
            self._decisions_fh.flush()

    def log_feedback(self, record: dict[str, Any]) -> None:
        """Append a feedback record to ``routing_feedback.jsonl``."""
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._feedback_fh.write(line)
            self._feedback_fh.write("\n")
            self._feedback_fh.flush()

    def close(self) -> None:
        """Flush and close file handles."""
        with self._lock:
            for fh in (self._decisions_fh, self._feedback_fh):
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
        logger.info("ReplayLogger closed (run_id=%s)", self.run_id)

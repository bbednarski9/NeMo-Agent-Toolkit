#!/usr/bin/env python3
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
Split an Agent Leaderboard v2 dataset file into subsets.

Modes
-----
--first-n-samples N
    Take the first N scenarios from the input file (preserving original order).
    Output: {stem}_first{N}.json

--split-by-percentages P1 P2 ... Pn  (must sum to 100)
    Split the dataset into len(P) disjoint, exhaustive partitions according to
    the given percentage weights.  Uses a deterministic shuffle (seed derived
    from the input filename) so that repeated runs produce identical splits.
    Outputs: {stem}_split0.json, {stem}_split1.json, ...

Examples
--------
# First 20 scenarios from banking (in original order)
python split_agent_leaderboard_v2.py \\
    --input-file data/agent_leaderboard_v2_banking.json \\
    --first-n-samples 20

# 20/80 split for banking (shuffled, deterministic)
python split_agent_leaderboard_v2.py \\
    --input-file data/agent_leaderboard_v2_banking.json \\
    --split-by-percentages 20 80

# 20/80 split, no shuffle, custom output prefix
python split_agent_leaderboard_v2.py \\
    --input-file data/agent_leaderboard_v2_banking.json \\
    --split-by-percentages 20 80 \\
    --no-shuffle \\
    --output-prefix data/banking_custom
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    logger.info("Loading %s", path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array at the top level.")
    logger.info("  %d scenarios loaded", len(data))
    return data


def _save(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("  Saved %d scenarios -> %s", len(data), path)


def _summary(data: list[dict]) -> str:
    ids = [s.get("id", "?") for s in data]
    sample = ", ".join(ids[:3]) + ("..." if len(ids) > 3 else "")
    return f"ids=[{sample}]"


# ---------------------------------------------------------------------------
# Mode: first-n-samples
# ---------------------------------------------------------------------------

def first_n_samples(input_file: Path, n: int, output_prefix: str | None) -> Path:
    data = _load(input_file)
    if n > len(data):
        logger.warning("--first-n-samples %d > dataset size %d; using all", n, len(data))
        n = len(data)

    subset = data[:n]
    stem = output_prefix if output_prefix else str(input_file.with_suffix(""))
    out = Path(f"{stem}_first{n}of{len(data)}.json")

    logger.info("first-n-samples: taking first %d of %d  %s", n, len(data), _summary(subset))
    _save(subset, out)
    return out


# ---------------------------------------------------------------------------
# Mode: split-by-percentages
# ---------------------------------------------------------------------------

def split_by_percentages(
    input_file: Path,
    percentages: list[int],
    shuffle: bool,
    output_prefix: str | None,
) -> list[Path]:
    total_pct = sum(percentages)
    if total_pct != 100:
        raise ValueError(
            f"Percentages must sum to 100, got {percentages} = {total_pct}"
        )

    data = _load(input_file)
    n = len(data)

    # Deterministic shuffle keyed on the stem so the same file always
    # produces the same split regardless of Python version.
    indices = list(range(n))
    if shuffle:
        seed = abs(hash(input_file.stem)) % (2**31)
        random.seed(seed)
        random.shuffle(indices)
        logger.info("Shuffled with seed %d (derived from %r)", seed, input_file.stem)
    else:
        logger.info("No shuffle — preserving original order")

    # Compute split boundaries using exact rounding to guarantee exhaustiveness.
    # Each split gets floor(p/100 * n) rows; remainder rows go to earlier splits.
    raw_sizes = [p / 100 * n for p in percentages]
    sizes = [int(s) for s in raw_sizes]
    remainder = n - sum(sizes)
    for i in range(remainder):
        sizes[i] += 1

    out_paths: list[Path] = []
    cursor = 0
    for i, size in enumerate(sizes):
        split_indices = indices[cursor: cursor + size]
        split_data = [data[j] for j in split_indices]
        cursor += size

        stem = output_prefix if output_prefix else str(input_file.with_suffix(""))
        out = Path(f"{stem}_split{i}_pct{percentages[i]}of100.json")

        logger.info(
            "split%d (%d%%): %d scenarios  %s",
            i, percentages[i], len(split_data), _summary(split_data),
        )
        _save(split_data, out)
        out_paths.append(out)

    # Sanity check
    assert cursor == n, f"Bug: processed {cursor}/{n} scenarios"
    total_out = sum(len(json.load(open(p))) for p in out_paths)
    assert total_out == n, f"Bug: output has {total_out}/{n} scenarios"
    logger.info("Split complete: %d scenarios -> %d files, no overlap, no loss.", n, len(out_paths))
    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Split an Agent Leaderboard v2 JSON dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-file", "-i",
        type=Path,
        required=True,
        help="Path to the source dataset JSON file (array of scenario objects).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Override the output path prefix.  If omitted the prefix is derived "
            "from --input-file by stripping its suffix."
        ),
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        default=False,
        help=(
            "For --split-by-percentages: preserve original order instead of "
            "applying a deterministic shuffle before splitting."
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--first-n-samples",
        type=int,
        metavar="N",
        help="Take the first N scenarios from the input file.",
    )
    mode.add_argument(
        "--split-by-percentages",
        type=int,
        nargs="+",
        metavar="PCT",
        help=(
            "Space-separated list of integer percentages that sum to 100.  "
            "Creates one output file per percentage."
        ),
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        parser.error(f"Input file not found: {args.input_file}")

    if args.first_n_samples is not None:
        if args.first_n_samples < 1:
            parser.error("--first-n-samples must be >= 1")
        first_n_samples(args.input_file, args.first_n_samples, args.output_prefix)

    elif args.split_by_percentages is not None:
        pcts = args.split_by_percentages
        if any(p <= 0 for p in pcts):
            parser.error("All percentages must be positive integers.")
        if sum(pcts) != 100:
            parser.error(f"Percentages must sum to 100, got {pcts} = {sum(pcts)}")
        split_by_percentages(
            args.input_file,
            pcts,
            shuffle=not args.no_shuffle,
            output_prefix=args.output_prefix,
        )


if __name__ == "__main__":
    main()

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
Create filtered test subsets from the full Agent Leaderboard v2 dataset.

Modes:
  Single:     Select N scenarios from one dataset (original behavior)
  Partitioned: Split a dataset into K non-overlapping partitions for
               multi-variant benchmarks where each variant gets unique inputs.

Examples:
  # Single subset (original)
  python create_test_subset.py --input-file data/banking.json --num-scenarios 5

  # Partitioned for 5 variants (20 scenarios each from 100)
  python create_test_subset.py --partitions 5 --domains banking healthcare insurance investment telecom
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def create_test_subset(input_file: Path, output_file: Path, num_scenarios: int = 3) -> None:
    """Create a test subset with a limited number of scenarios."""
    logger.info("Loading full dataset from %s", input_file)

    with open(input_file) as f:
        full_dataset = json.load(f)

    logger.info("Loaded %d scenarios from full dataset", len(full_dataset))

    test_subset = full_dataset[:num_scenarios]

    logger.info("Created test subset with %d scenarios", len(test_subset))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(test_subset, f, indent=2)

    logger.info("Saved test subset to %s", output_file)

    for i, scenario in enumerate(test_subset):
        logger.info(
            "Scenario %d: id=%s, goals=%d, tools=%d, expected_calls=%d",
            i + 1,
            scenario.get("id"),
            len(scenario.get("user_goals", [])),
            len(scenario.get("available_tools", [])),
            len(scenario.get("expected_tool_calls", [])),
        )


def create_partitioned_subsets(
    domains: list[str],
    num_partitions: int,
    data_dir: Path,
    shuffle: bool = True,
) -> dict[str, list[Path]]:
    """Split each domain's full dataset into non-overlapping partitions.

    Returns a dict mapping domain -> list of output file paths (one per partition).
    """
    import random

    results: dict[str, list[Path]] = {}

    for domain in domains:
        input_file = data_dir / f"agent_leaderboard_v2_{domain}.json"
        if not input_file.exists():
            logger.warning("Skipping %s: %s not found", domain, input_file)
            continue

        with open(input_file) as f:
            full_dataset = json.load(f)

        total = len(full_dataset)
        logger.info("%s: %d total scenarios, splitting into %d partitions", domain, total, num_partitions)

        indices = list(range(total))
        if shuffle:
            random.seed(42 + hash(domain))
            random.shuffle(indices)

        partition_size = total // num_partitions
        output_files = []

        for vi in range(num_partitions):
            start = vi * partition_size
            end = start + partition_size if vi < num_partitions - 1 else total
            partition_indices = indices[start:end]
            partition_data = [full_dataset[i] for i in partition_indices]

            output_file = data_dir / f"agent_leaderboard_v2_{domain}_v{vi}.json"
            with open(output_file, "w") as f:
                json.dump(partition_data, f, indent=2)

            output_files.append(output_file)
            logger.info("  %s_v%d: %d scenarios -> %s", domain, vi, len(partition_data), output_file)

        results[domain] = output_files

    return results


def update_configs(config_dir: Path, data_dir: Path, domains: list[str], num_partitions: int) -> None:
    """Update generated config YAMLs to point to their partitioned dataset files."""
    import yaml

    for domain in domains:
        for vi in range(num_partitions):
            config_path = config_dir / f"{domain}_v{vi}.yml"
            if not config_path.exists():
                logger.warning("Config not found: %s", config_path)
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f)

            data_file = f"./examples/dynamo_integration/data/agent_leaderboard_v2_{domain}_v{vi}.json"
            config["eval"]["general"]["dataset"]["file_path"] = data_file

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, width=120, sort_keys=False)

            logger.info("Updated %s -> %s", config_path.name, data_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test subsets from Agent Leaderboard v2 dataset")

    sub = parser.add_subparsers(dest="mode", help="Operation mode")

    single = sub.add_parser("single", help="Create a single test subset (original behavior)")
    single.add_argument("--input-file", type=Path,
                        default=DATA_DIR / "agent_leaderboard_v2_banking.json")
    single.add_argument("--output-file", type=Path,
                        default=DATA_DIR / "agent_leaderboard_v2_test_subset.json")
    single.add_argument("--num-scenarios", type=int, default=3)

    part = sub.add_parser("partition", help="Split datasets into non-overlapping partitions for multi-variant configs")
    part.add_argument("--domains", nargs="+",
                      default=["banking", "healthcare", "insurance", "investment", "telecom"])
    part.add_argument("--partitions", type=int, default=5, help="Number of partitions per domain")
    part.add_argument("--data-dir", type=Path, default=DATA_DIR)
    part.add_argument("--config-dir", type=Path, default=None,
                      help="If set, update config YAMLs to point to partitioned datasets")
    part.add_argument("--no-shuffle", action="store_true", help="Don't shuffle before partitioning")

    args = parser.parse_args()

    if args.mode == "single" or args.mode is None:
        if args.mode is None:
            args.input_file = DATA_DIR / "agent_leaderboard_v2_banking.json"
            args.output_file = DATA_DIR / "agent_leaderboard_v2_test_subset.json"
            args.num_scenarios = 3
        create_test_subset(args.input_file, args.output_file, args.num_scenarios)
    elif args.mode == "partition":
        create_partitioned_subsets(args.domains, args.partitions, args.data_dir, shuffle=not args.no_shuffle)
        if args.config_dir:
            update_configs(args.config_dir, args.data_dir, args.domains, args.partitions)

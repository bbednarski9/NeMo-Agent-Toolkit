#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate multi-prompt config variants for KV cache diversity benchmarking.

Takes existing per-domain YAML configs and produces N variants per domain,
each with a unique prefix paragraph prepended to the system prompt. This
forces KV cache misses between variants (different prefix = different blocks),
simulating a realistic multi-agent environment where different agent instances
use different system prompts.

Usage:
    python generate_multi_prompt_configs.py \
        --source-dir configs/multi_domain_rethinking \
        --output-dir configs/multi_domain_25prompt \
        --variants-per-domain 5

This produces 25 configs (5 domains × 5 variants) with output dirs like:
    .../multi_domain_25prompt/banking_v0/
    .../multi_domain_25prompt/banking_v1/
    ...
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml

CANNED_RESPONSES = [
    "Successfully executed {tool_name}. Operation completed.",
    "Completed {tool_name} operation. All parameters validated and action confirmed.",
    "Action {tool_name} finished. Result received and verified successfully.",
    "{tool_name} executed without errors. Response data has been processed.",
    "Finished running {tool_name}. The system confirmed the operation was successful.",
]

PERSONA_PREFIXES = [
    (
        "You are Agent Instance Alpha. Your operational ID is ALPHA-{domain}-001. "
        "You prioritize thoroughness and always verify each step before proceeding. "
        "When uncertain, you gather additional information rather than making assumptions. "
        "Your approach is methodical: gather context, plan steps, execute, verify results.\n\n"
    ),
    (
        "You are Agent Instance Bravo. Your operational ID is BRAVO-{domain}-002. "
        "You prioritize efficiency and aim to resolve requests in the fewest steps possible. "
        "You leverage your domain expertise to skip unnecessary verification steps when confident. "
        "Your approach is direct: identify the core need, execute the minimum required steps, confirm completion.\n\n"
    ),
    (
        "You are Agent Instance Charlie. Your operational ID is CHARLIE-{domain}-003. "
        "You prioritize user communication and always explain your reasoning before taking action. "
        "You break complex requests into clearly labeled phases and confirm understanding before each phase. "
        "Your approach is consultative: understand intent, propose plan, execute with narration, summarize.\n\n"
    ),
    (
        "You are Agent Instance Delta. Your operational ID is DELTA-{domain}-004. "
        "You prioritize risk mitigation and always consider edge cases and potential failures. "
        "You proactively check preconditions before executing actions and have fallback plans ready. "
        "Your approach is defensive: assess risks, validate preconditions, execute with safeguards, audit results.\n\n"
    ),
    (
        "You are Agent Instance Echo. Your operational ID is ECHO-{domain}-005. "
        "You prioritize completeness and always explore related opportunities beyond the explicit request. "
        "You suggest additional services or actions that could benefit the user based on their context. "
        "Your approach is proactive: address the request, identify related needs, offer enhancements, close comprehensively.\n\n"
    ),
]


def generate_variants(source_dir: Path, output_dir: Path, variants_per_domain: int):
    import random as _random

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    configs = sorted(source_dir.glob("*.yml"))
    if not configs:
        print(f"ERROR: No .yml files found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    variants_per_domain = min(variants_per_domain, len(PERSONA_PREFIXES))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_variant_names = []

    for config_path in configs:
        domain = config_path.stem
        with open(config_path) as f:
            base_config = yaml.safe_load(f)

        # system_prompt location varies:
        #   rethinking: functions.react_workflow.system_prompt
        #   no-rethinking: workflow.system_prompt
        if "react_workflow" in base_config.get("functions", {}):
            prompt_path = ("functions", "react_workflow", "system_prompt")
        elif "system_prompt" in base_config.get("workflow", {}):
            prompt_path = ("workflow", "system_prompt")
        else:
            print(f"WARNING: no system_prompt found in {config_path}, skipping", file=sys.stderr)
            continue

        obj = base_config
        for k in prompt_path:
            obj = obj[k]
        original_prompt = obj
        base_output_dir = base_config["eval"]["general"]["output"]["dir"]

        # Find the tool include list to shuffle per variant
        tool_group_key = None
        original_tool_order = None
        for fg_name, fg_val in base_config.get("function_groups", {}).items():
            if isinstance(fg_val, dict) and "include" in fg_val:
                tool_group_key = fg_name
                original_tool_order = list(fg_val["include"])
                break

        for vi in range(variants_per_domain):
            variant_name = f"{domain}_v{vi}"
            all_variant_names.append(variant_name)

            config = copy.deepcopy(base_config)

            # 1. Unique system prompt prefix (diverges at token 0)
            persona = PERSONA_PREFIXES[vi].format(domain=domain.upper())
            obj = config
            for k in prompt_path[:-1]:
                obj = obj[k]
            obj[prompt_path[-1]] = persona + original_prompt

            # 2. Create a shuffled tools.json for this variant so that tool
            #    definitions are registered (and thus rendered in the prompt)
            #    in a different order.  The include list order is lost to a
            #    set() in NAT's FunctionGroup, so we must shuffle the source.
            if tool_group_key:
                src_tools_path = base_config["function_groups"][tool_group_key].get("tools_json_path", "")
                if src_tools_path:
                    abs_src = Path(src_tools_path)
                    if not abs_src.exists():
                        abs_src = Path(__file__).parent.parent / src_tools_path
                    if abs_src.exists():
                        with open(abs_src) as tf:
                            tool_schemas = json.load(tf)
                        rng = _random.Random(42 + hash(domain) + vi)
                        rng.shuffle(tool_schemas)
                        variant_tools_dir = output_dir / "tools"
                        variant_tools_dir.mkdir(parents=True, exist_ok=True)
                        variant_tools_path = variant_tools_dir / f"{domain}_v{vi}_tools.json"
                        with open(variant_tools_path, "w") as tf:
                            json.dump(tool_schemas, tf, indent=2)
                        config["function_groups"][tool_group_key]["tools_json_path"] = str(variant_tools_path)

            # 3. Unique canned tool response (different scratchpad tokens per turn)
            config["functions"]["react_benchmark_agent"]["canned_response_template"] = \
                CANNED_RESPONSES[vi % len(CANNED_RESPONSES)]

            # 4. Unique prefix_id_template for router radix tree tracking
            config["llms"]["dynamo_llm"]["nvext_prefix_id_template"] = f"{domain}-v{vi}-{{uuid}}"

            # Output to a variant-specific directory
            parent = str(Path(base_output_dir).parent)
            config["eval"]["general"]["output"]["dir"] = f"{parent}/{variant_name}/"

            variant_path = output_dir / f"{variant_name}.yml"
            with open(variant_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, width=120, sort_keys=False)

            tool_info = f", tools shuffled (seed={42 + hash(domain) + vi})" if original_tool_order else ""
            print(f"  Generated: {variant_path}{tool_info}")

    print(f"\nGenerated {len(all_variant_names)} configs ({len(configs)} domains × {variants_per_domain} variants)")
    print(f"Variance sources per variant:")
    print(f"  1. Unique persona prefix (system prompt diverges at token 0)")
    print(f"  2. Shuffled tool order (different {'{tools}'} expansion, ~5000 tokens)")
    print(f"  3. Unique canned tool response template (different scratchpad tokens)")
    print(f"  4. Unique dataset partition (different user inputs, via create_test_subset.py)")
    print(f"Output directory: {output_dir}")

    print(f"\nFor run_multi_domain_benchmark.sh, use:")
    print(f'  DOMAINS="{" ".join(all_variant_names)}"')
    print(f'  CONFIG_DIR="{output_dir}"')


def main():
    parser = argparse.ArgumentParser(description="Generate multi-prompt config variants")
    parser.add_argument("--source-dir", required=True, help="Directory with base domain configs")
    parser.add_argument("--output-dir", required=True, help="Directory for generated variant configs")
    parser.add_argument("--variants-per-domain", type=int, default=5, help="Number of prompt variants per domain (max 5)")
    args = parser.parse_args()

    generate_variants(Path(args.source_dir), Path(args.output_dir), args.variants_per_domain)


if __name__ == "__main__":
    main()

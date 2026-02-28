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
Domain Tools Registration for Agent Leaderboard v2 Multi-Domain Evaluation.

Registers tool groups for healthcare, insurance, investment, and telecom domains.
Banking is already registered in banking_tools.py.
"""

import json
import logging
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

from .tool_intent_stubs import ToolIntentBuffer
from .tool_intent_stubs import create_tool_stub_function

logger = logging.getLogger(__name__)


def _register_domain_tools(config, builder: Builder):
    """Shared logic for registering domain tool stubs from a tools.json file."""
    group = FunctionGroup(config=config)

    if not config.decision_only:
        logger.info("decision_only is False, skipping %s tools stub registration", config.__class__.__name__)
        return group

    if not hasattr(builder, "runtime_metadata"):
        builder.runtime_metadata = {}

    intent_buffer = builder.runtime_metadata.get("tool_intent_buffer")
    if intent_buffer is None:
        intent_buffer = ToolIntentBuffer()
        builder.runtime_metadata["tool_intent_buffer"] = intent_buffer

    tools_path = Path(__file__).parent / config.tools_json_path
    if not tools_path.exists():
        tools_path = Path(config.tools_json_path)
    if not tools_path.exists():
        raise FileNotFoundError(f"Tools file not found: {tools_path}")

    with open(tools_path) as f:
        tools_schemas = json.load(f)

    logger.info("Loaded %d tool schemas from %s", len(tools_schemas), tools_path)

    registered_count = 0
    for tool_schema in tools_schemas:
        tool_name = tool_schema.get("title", "")
        if not tool_name:
            continue
        try:
            stub_fn, custom_input_schema, description = create_tool_stub_function(tool_schema, intent_buffer)
            group.add_function(name=tool_name, fn=stub_fn, input_schema=custom_input_schema, description=description)
            registered_count += 1
        except Exception:
            logger.exception("Failed to add tool stub for %s", tool_name)
            continue

    logger.info("Registered %d/%d tool stubs for %s", registered_count, len(tools_schemas), config.__class__.__name__)
    return group


# ---------------------------------------------------------------------------
# Healthcare
# ---------------------------------------------------------------------------
class HealthcareToolsGroupConfig(FunctionGroupBaseConfig, name="healthcare_tools_group"):
    """Configuration for loading healthcare tools as a function group."""

    tools_json_path: str = Field(
        default="data/raw/healthcare/tools.json",
        description="Path to tools.json for healthcare domain",
    )
    decision_only: bool = Field(default=True)


@register_function_group(config_type=HealthcareToolsGroupConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def healthcare_tools_group_function(config: HealthcareToolsGroupConfig, builder: Builder):
    yield _register_domain_tools(config, builder)


# ---------------------------------------------------------------------------
# Insurance
# ---------------------------------------------------------------------------
class InsuranceToolsGroupConfig(FunctionGroupBaseConfig, name="insurance_tools_group"):
    """Configuration for loading insurance tools as a function group."""

    tools_json_path: str = Field(
        default="data/raw/insurance/tools.json",
        description="Path to tools.json for insurance domain",
    )
    decision_only: bool = Field(default=True)


@register_function_group(config_type=InsuranceToolsGroupConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def insurance_tools_group_function(config: InsuranceToolsGroupConfig, builder: Builder):
    yield _register_domain_tools(config, builder)


# ---------------------------------------------------------------------------
# Investment
# ---------------------------------------------------------------------------
class InvestmentToolsGroupConfig(FunctionGroupBaseConfig, name="investment_tools_group"):
    """Configuration for loading investment tools as a function group."""

    tools_json_path: str = Field(
        default="data/raw/investment/tools.json",
        description="Path to tools.json for investment domain",
    )
    decision_only: bool = Field(default=True)


@register_function_group(config_type=InvestmentToolsGroupConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def investment_tools_group_function(config: InvestmentToolsGroupConfig, builder: Builder):
    yield _register_domain_tools(config, builder)


# ---------------------------------------------------------------------------
# Telecom
# ---------------------------------------------------------------------------
class TelecomToolsGroupConfig(FunctionGroupBaseConfig, name="telecom_tools_group"):
    """Configuration for loading telecom tools as a function group."""

    tools_json_path: str = Field(
        default="data/raw/telecom/tools.json",
        description="Path to tools.json for telecom domain",
    )
    decision_only: bool = Field(default=True)


@register_function_group(config_type=TelecomToolsGroupConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def telecom_tools_group_function(config: TelecomToolsGroupConfig, builder: Builder):
    yield _register_domain_tools(config, builder)

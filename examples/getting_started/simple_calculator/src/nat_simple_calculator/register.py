# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

logger = logging.getLogger(__name__)

# pylint: disable=unused-argument


def extract_numbers(text: str) -> list[str]:
    """
    Extract numerical values (including floats) from text.

    Args:
        text: Input text containing numbers

    Returns:
        List of number strings found in the text
    """
    import re
    return re.findall(r"\d+(?:\.\d+)?", text)


def validate_number_count(numbers: list[str], expected_count: int, action: str) -> str | None:
    if len(numbers) < expected_count:
        return f"Provide at least {expected_count} numbers to {action}."
    if len(numbers) > expected_count:
        return f"This tool only supports {action} between {expected_count} numbers."
    return None


class CalculatorToolConfig(FunctionGroupBaseConfig, name="calculator"):
    include: list[str] = Field(
        description="The functions to include in the calculator function group.",
        default_factory=lambda: ["add", "subtract", "multiply", "divide", "compare"],
    )


@register_function_group(config_type=CalculatorToolConfig)
async def calculator_group(config: CalculatorToolConfig, _builder: Builder):

    async def _multiply(text: str) -> str:
        """
        This is a mathematical tool used to multiply two numbers together.
        It takes 2 numbers as an input and computes their numeric product as the output.
        """
        numbers = extract_numbers(text)
        validation_error = validate_number_count(numbers, expected_count=2, action="multiply")
        if validation_error:
            return validation_error
        a = float(numbers[0])
        b = float(numbers[1])
        return f"The product of {a} * {b} is {a * b}"

    async def _add(text: str) -> str:
        """
        This is a mathematical tool used to add two numbers together.
        It takes 2 numbers as an input and computes their numeric sum as the output.
        """
        numbers = extract_numbers(text)
        validation_error = validate_number_count(numbers, expected_count=2, action="add")
        if validation_error:
            return validation_error
        a = float(numbers[0])
        b = float(numbers[1])
        return f"The sum of {a} + {b} is {a + b}"

    async def _divide(text: str) -> str:
        """
        This is a mathematical tool used to divide one number by another.
        It takes 2 numbers as an input and computes their numeric quotient as the output.
        """
        numbers = extract_numbers(text)
        validation_error = validate_number_count(numbers, expected_count=2, action="divide")
        if validation_error:
            return validation_error
        a = float(numbers[0])
        b = float(numbers[1])
        return f"The result of {a} / {b} is {a / b}"

    async def _subtract(text: str) -> str:
        """
        This is a mathematical tool used to subtract one number from another.
        It takes 2 numbers as an input and computes their numeric difference as the output.
        """
        numbers = extract_numbers(text)
        validation_error = validate_number_count(numbers, expected_count=2, action="subtract")
        if validation_error:
            return validation_error
        a = float(numbers[0])
        b = float(numbers[1])
        return f"The result of {a} - {b} is {a - b}"

    async def _compare(text: str) -> str:
        """
        This is a mathematical tool used to compare two numbers.
        It takes 2 numbers as an input and determines if one is greater, less than, or equal to the other.
        """
        numbers = extract_numbers(text)
        validation_error = validate_number_count(numbers, expected_count=2, action="compare")
        if validation_error:
            return validation_error
        a = float(numbers[0])
        b = float(numbers[1])
        if a > b:
            return f"The first number {a} is greater than the second number {b}"
        if a < b:
            return f"The first number {a} is less than the second number {b}"
        return f"The first number {a} is equal to the second number {b}"

    group = FunctionGroup(config=config)
    group.add_function("add", _add, description=_add.__doc__)
    group.add_function("subtract", _subtract, description=_subtract.__doc__)
    group.add_function("multiply", _multiply, description=_multiply.__doc__)
    group.add_function("divide", _divide, description=_divide.__doc__)
    group.add_function("compare", _compare, description=_compare.__doc__)
    yield group

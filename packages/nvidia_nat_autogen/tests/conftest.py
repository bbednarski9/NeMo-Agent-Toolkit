# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Conftest for testing """
import sys
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_autogen_imports() -> dict:
    """Mock AutoGen imports for testing.

    Yields:
        dict: Dictionary of mocked AutoGen modules.
    """
    # Mock autogen modules
    autogen_core = Mock()
    autogen_core.tools = Mock()
    autogen_core.models = Mock()

    autogen_ext = Mock()
    autogen_ext.models = Mock()
    autogen_ext.models.openai = Mock()

    # Add specific mocks for commonly used classes
    autogen_ext.models.ModelInfo = Mock()
    autogen_ext.models.SystemMessage = Mock()

    sys.modules['autogen_core'] = autogen_core
    sys.modules['autogen_core.tools'] = autogen_core.tools
    sys.modules['autogen_core.models'] = autogen_core.models
    sys.modules['autogen_ext'] = autogen_ext
    sys.modules['autogen_ext.models'] = autogen_ext.models
    sys.modules['autogen_ext.models.openai'] = autogen_ext.models.openai

    yield {'autogen_core': autogen_core, 'autogen_ext': autogen_ext}

    # Clean up
    modules_to_remove = [
        'autogen_core',
        'autogen_core.tools',
        'autogen_core.models',
        'autogen_ext',
        'autogen_ext.models',
        'autogen_ext.models.openai'
    ]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]

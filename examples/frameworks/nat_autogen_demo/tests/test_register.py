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

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.data_models.component_ref import LLMRef
from nat_autogen_demo.register import AutoGenFunctionConfig


class TestAutoGenFunctionConfig:
    """Test cases for AutoGenFunctionConfig class."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref)

        assert config.llm_name == llm_ref
        assert config.tool_names == []
        assert config.type == "autogen_team"

    def test_config_creation_with_tools(self):
        """Test creating config with tool names."""
        llm_ref = LLMRef("test_llm")
        tools = ["tool1", "tool2", "tool3"]
        config = AutoGenFunctionConfig(llm_name=llm_ref, tool_names=tools)

        assert config.llm_name == llm_ref
        assert config.tool_names == tools

    def test_config_validation(self):
        """Test config field validation."""
        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref, tool_names=["tool1"])

        # Test that the config is properly validated
        assert isinstance(config.llm_name, LLMRef)
        assert isinstance(config.tool_names, list)
        assert all(isinstance(tool, str) for tool in config.tool_names)

    def test_config_name_attribute(self):
        """Test config name attribute."""
        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref)

        # Check that the config has the correct name
        assert config.type == "autogen_team"
        assert config.full_type == "nat_autogen_demo/autogen_team"

    def test_config_field_types(self):
        """Test config field types."""
        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref, tool_names=["tool1", "tool2"])

        # Validate field types
        assert isinstance(config.llm_name, LLMRef)
        assert str(config.llm_name) == "test_llm"
        assert isinstance(config.tool_names, list)
        assert len(config.tool_names) == 2

    def test_config_empty_tools_list(self):
        """Test config with empty tools list."""
        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref, tool_names=[])

        assert config.tool_names == []

    def test_config_multiple_tools(self):
        """Test config with multiple tools."""
        llm_ref = LLMRef("test_llm")
        tools = ["weather_tool", "time_tool", "calculator_tool"]
        config = AutoGenFunctionConfig(llm_name=llm_ref, tool_names=tools)

        assert config.tool_names == tools
        assert len(config.tool_names) == 3

    def test_llm_ref_component_group(self):
        """Test LLMRef component group."""
        llm_ref = LLMRef("test_llm")

        # Test the LLMRef has correct component group
        from nat.data_models.component import ComponentGroup
        assert llm_ref.component_group == ComponentGroup.LLMS

    def test_config_inheritance(self):
        """Test that AutoGenFunctionConfig inherits from FunctionBaseConfig."""
        from nat.data_models.function import FunctionBaseConfig

        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref)

        assert isinstance(config, FunctionBaseConfig)

    def test_config_field_descriptions(self):
        """Test that config fields have proper descriptions."""
        # Check that model fields exist
        model_fields = AutoGenFunctionConfig.model_fields
        assert "llm_name" in model_fields
        assert "tool_names" in model_fields

        # Check descriptions
        assert model_fields["llm_name"].description == "The LLM model to use with AutoGen agents."
        assert model_fields["tool_names"].description == "List of tool names to be used by the agents."


class TestAutoGenTeamFunction:
    """Test cases for autogen_team function."""

    def test_autogen_team_function_exists(self):
        """Test that autogen_team function exists and is callable."""
        from nat_autogen_demo.register import autogen_team

        assert callable(autogen_team)

    def test_autogen_team_has_decorator(self):
        """Test that autogen_team function has been decorated."""
        from nat_autogen_demo.register import autogen_team

        # The function should be wrapped by the decorator
        assert hasattr(autogen_team, '__wrapped__') or callable(autogen_team)

    def test_autogen_team_docstring(self):
        """Test that autogen_team function has proper docstring."""
        from nat_autogen_demo.register import autogen_team

        assert autogen_team.__doc__ is not None
        assert "AutoGen multi-agent workflow" in autogen_team.__doc__
        assert "collaborative agents" in autogen_team.__doc__

    def test_autogen_team_function_signature(self):
        """Test autogen_team function signature."""
        import inspect

        from nat_autogen_demo.register import autogen_team

        sig = inspect.signature(autogen_team)
        params = list(sig.parameters.keys())

        assert "config" in params
        assert "builder" in params

    def test_function_annotations(self):
        """Test function type annotations."""
        import inspect

        from nat_autogen_demo.register import autogen_team

        sig = inspect.signature(autogen_team)

        # Check parameter annotations
        assert sig.parameters['config'].annotation.__name__ == 'AutoGenFunctionConfig'
        assert 'Builder' in str(sig.parameters['builder'].annotation)


class TestModuleImports:
    """Test module imports and structure."""

    def test_required_imports(self):
        """Test that all required imports are available."""
        # Test that we can import all the required modules
        from nat_autogen_demo.register import AutoGenFunctionConfig
        from nat_autogen_demo.register import autogen_team
        from nat_autogen_demo.register import logger

        assert logger is not None
        assert AutoGenFunctionConfig is not None
        assert autogen_team is not None

    def test_logger_configuration(self):
        """Test logger configuration."""
        from nat_autogen_demo.register import logger

        assert isinstance(logger, logging.Logger)
        assert logger.name == 'nat_autogen_demo.register'

    def test_imports_from_nat(self):
        """Test imports from NAT framework."""
        # Test that NAT imports work
        from nat.builder.framework_enum import LLMFrameworkEnum
        from nat.builder.function_info import FunctionInfo
        from nat.data_models.component_ref import LLMRef
        from nat.data_models.function import FunctionBaseConfig

        assert LLMFrameworkEnum is not None
        assert FunctionInfo is not None
        assert LLMRef is not None
        assert FunctionBaseConfig is not None

    def test_autogen_imports_exist(self):
        """Test that AutoGen imports are structured correctly in the function."""
        import ast
        import inspect

        from nat_autogen_demo.register import autogen_team

        # Get the source code
        source = inspect.getsource(autogen_team)

        # Parse the AST to check for imports
        tree = ast.parse(source)

        # Check that the function contains import statements
        has_imports = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                has_imports = True
                break

        assert has_imports


class TestConstants:
    """Test constants and hardcoded values."""

    def test_mcp_server_url(self):
        """Test MCP server URL configuration."""
        import ast
        import inspect

        from nat_autogen_demo.register import autogen_team

        source = inspect.getsource(autogen_team)
        tree = ast.parse(source)

        # Look for the MCP URL
        found_url = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "http://0.0.0.0:9901/mcp" in node.value:
                    found_url = True
                    break

        assert found_url

    def test_agent_names(self):
        """Test agent names are defined correctly."""
        import ast
        import inspect

        from nat_autogen_demo.register import autogen_team

        source = inspect.getsource(autogen_team)
        tree = ast.parse(source)

        # Look for agent names
        found_weather_agent = False
        found_final_agent = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "WeatherAndTimeAgent" in node.value:
                    found_weather_agent = True
                elif "FinalResponseAgent" in node.value:
                    found_final_agent = True

        assert found_weather_agent
        assert found_final_agent

    def test_termination_words(self):
        """Test termination condition words."""
        import ast
        import inspect

        from nat_autogen_demo.register import autogen_team

        source = inspect.getsource(autogen_team)
        tree = ast.parse(source)

        # Look for termination words
        found_done = False
        found_approve = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "DONE" in node.value:
                    found_done = True
                elif "APPROVE" in node.value:
                    found_approve = True

        assert found_done
        assert found_approve


class TestErrorScenarios:
    """Test error handling scenarios."""

    def test_config_with_invalid_llm_ref(self):
        """Test config creation with invalid LLM ref."""
        # Test that we can create LLMRef with string
        llm_ref = LLMRef("invalid_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref)

        assert config.llm_name == llm_ref

    def test_config_with_none_tools(self):
        """Test config behavior with None tools - should use default."""
        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref)

        # Should use default empty list
        assert config.tool_names == []

    def test_empty_tool_names(self):
        """Test config with empty tool names."""
        llm_ref = LLMRef("test_llm")
        config = AutoGenFunctionConfig(llm_name=llm_ref, tool_names=[])

        assert config.tool_names == []


class TestAutoGenTeamFunctionImplementation:
    """Comprehensive tests for the autogen_team function implementation."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AutoGenFunctionConfig."""
        llm_ref = LLMRef("test_llm")
        return AutoGenFunctionConfig(llm_name=llm_ref, tool_names=["test_tool"])

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder."""
        builder = AsyncMock(spec=Builder)
        builder.get_llm = AsyncMock()
        builder.get_tools = AsyncMock(return_value=[])
        return builder

    @pytest.mark.asyncio
    async def test_autogen_team_successful_execution(self, mock_config, mock_builder):
        """Test successful execution of autogen_team function."""
        with patch('autogen_agentchat.agents.AssistantAgent') as mock_agent, \
             patch('autogen_agentchat.teams.RoundRobinGroupChat') as mock_group_chat, \
             patch('autogen_ext.tools.mcp.StreamableHttpMcpToolAdapter') as mock_adapter:

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = [AsyncMock()]

            # Setup MCP adapter
            mock_adapter_instance = AsyncMock()
            mock_adapter.from_server_params = AsyncMock(return_value=mock_adapter_instance)

            # Setup team
            mock_team_instance = AsyncMock()
            mock_group_chat.return_value = mock_team_instance

            # Setup result
            mock_result = MagicMock()
            mock_result.messages = [MagicMock()]
            mock_result.messages[-1].content = "Test response"
            mock_team_instance.run = AsyncMock(return_value=mock_result)

            # Import and test the function
            from nat_autogen_demo.register import autogen_team

            # Execute the function
            async with autogen_team(mock_config, mock_builder) as function_info:
                assert isinstance(function_info, FunctionInfo)
                assert function_info.single_fn is not None

                # Test the workflow function
                workflow_fn = function_info.single_fn
                result = await workflow_fn("What's the weather today?")
                assert result == "Test response"

                # Verify builder methods were called
                mock_builder.get_llm.assert_called_once_with(mock_config.llm_name,
                                                             wrapper_type=LLMFrameworkEnum.AUTOGEN)
                mock_builder.get_tools.assert_called_once_with(mock_config.tool_names,
                                                               wrapper_type=LLMFrameworkEnum.AUTOGEN)

                # Verify agents were created
                assert mock_agent.call_count == 2

    @pytest.mark.asyncio
    async def test_autogen_team_no_messages_in_result(self, mock_config, mock_builder):
        """Test when the AutoGen result has no messages."""
        with patch('autogen_agentchat.agents.AssistantAgent'), \
             patch('autogen_agentchat.teams.RoundRobinGroupChat') as mock_group_chat, \
             patch('autogen_ext.tools.mcp.StreamableHttpMcpToolAdapter') as mock_adapter, \
             patch('autogen_ext.tools.mcp.StreamableHttpServerParams'), \
             patch('autogen_agentchat.conditions.TextMentionTermination'):

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = [AsyncMock()]

            # Setup MCP adapter
            mock_adapter_instance = AsyncMock()
            mock_adapter.from_server_params = AsyncMock(return_value=mock_adapter_instance)

            # Setup team with no messages
            mock_team_instance = AsyncMock()
            mock_group_chat.return_value = mock_team_instance

            # Setup result with no messages
            mock_result = MagicMock()
            mock_result.messages = []
            mock_team_instance.run = AsyncMock(return_value=mock_result)

            # Execute the function
            from nat_autogen_demo.register import autogen_team
            async with autogen_team(mock_config, mock_builder) as function_info:
                workflow_fn = function_info.single_fn
                result = await workflow_fn("test input")
                assert result == "The workflow finished but no output was generated."

    @pytest.mark.asyncio
    async def test_autogen_team_no_messages_attribute(self, mock_config, mock_builder):
        """Test when the AutoGen result has no messages attribute."""
        with patch('autogen_agentchat.agents.AssistantAgent'), \
             patch('autogen_agentchat.teams.RoundRobinGroupChat') as mock_group_chat, \
             patch('autogen_ext.tools.mcp.StreamableHttpMcpToolAdapter') as mock_adapter, \
             patch('autogen_ext.tools.mcp.StreamableHttpServerParams'), \
             patch('autogen_agentchat.conditions.TextMentionTermination'):

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = [AsyncMock()]

            # Setup MCP adapter
            mock_adapter_instance = AsyncMock()
            mock_adapter.from_server_params = AsyncMock(return_value=mock_adapter_instance)

            # Setup team with result that has no messages attribute
            mock_team_instance = AsyncMock()
            mock_group_chat.return_value = mock_team_instance

            # Setup result without messages attribute - create a class without messages
            class ResultWithoutMessages:
                pass

            mock_result = ResultWithoutMessages()
            mock_team_instance.run = AsyncMock(return_value=mock_result)

            # Execute the function
            from nat_autogen_demo.register import autogen_team
            async with autogen_team(mock_config, mock_builder) as function_info:
                workflow_fn = function_info.single_fn
                result = await workflow_fn("test input")
                assert result == "The workflow finished but no output was generated."

    @pytest.mark.asyncio
    async def test_autogen_team_exception_handling(self, mock_config, mock_builder):
        """Test exception handling in the workflow."""
        with patch('autogen_agentchat.agents.AssistantAgent'), \
             patch('autogen_agentchat.teams.RoundRobinGroupChat') as mock_group_chat, \
             patch('autogen_ext.tools.mcp.StreamableHttpMcpToolAdapter') as mock_adapter, \
             patch('autogen_ext.tools.mcp.StreamableHttpServerParams'), \
             patch('autogen_agentchat.conditions.TextMentionTermination'):

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = [AsyncMock()]

            # Setup MCP adapter
            mock_adapter_instance = AsyncMock()
            mock_adapter.from_server_params = AsyncMock(return_value=mock_adapter_instance)

            # Setup team to raise an exception
            mock_team_instance = AsyncMock()
            mock_group_chat.return_value = mock_team_instance
            mock_team_instance.run = AsyncMock(side_effect=Exception("Test error"))

            # Execute the function
            from nat_autogen_demo.register import autogen_team
            async with autogen_team(mock_config, mock_builder) as function_info:
                workflow_fn = function_info.single_fn
                result = await workflow_fn("test input")
                assert result == "Error occurred during AutoGen workflow: Test error"

    @pytest.mark.asyncio
    async def test_autogen_team_workflow_no_messages(self, mock_config, mock_builder):
        """Test workflow when result has no messages."""
        with patch('nat_autogen_demo.register.AssistantAgent'), \
             patch('nat_autogen_demo.register.RoundRobinGroupChat') as mock_group_chat, \
             patch('nat_autogen_demo.register.StreamableHttpMcpToolAdapter'), \
             patch('nat_autogen_demo.register.StreamableHttpServerParams'), \
             patch('nat_autogen_demo.register.TextMentionTermination'):

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = []

            # Setup team with no messages
            mock_team_instance = AsyncMock()
            mock_group_chat.return_value = mock_team_instance

            mock_result = MagicMock()
            mock_result.messages = []
            mock_team_instance.run = AsyncMock(return_value=mock_result)

            from nat_autogen_demo.register import autogen_team

            async with autogen_team(mock_config, mock_builder) as function_info:
                workflow_fn = function_info.single_fn
                result = await workflow_fn("test input")
                assert result == "The workflow finished but no output was generated."

    @pytest.mark.asyncio
    async def test_autogen_team_workflow_no_messages_attribute(self, mock_config, mock_builder):
        """Test workflow when result has no messages attribute."""
        with patch('nat_autogen_demo.register.AssistantAgent'), \
             patch('nat_autogen_demo.register.RoundRobinGroupChat') as mock_group_chat, \
             patch('nat_autogen_demo.register.StreamableHttpMcpToolAdapter'), \
             patch('nat_autogen_demo.register.StreamableHttpServerParams'), \
             patch('nat_autogen_demo.register.TextMentionTermination'):

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = []

            # Setup team with result that has no messages attribute
            mock_team_instance = AsyncMock()
            mock_group_chat.return_value = mock_team_instance

            mock_result = MagicMock()
            del mock_result.messages  # Remove messages attribute
            mock_team_instance.run = AsyncMock(return_value=mock_result)

            from nat_autogen_demo.register import autogen_team

            async with autogen_team(mock_config, mock_builder) as function_info:
                workflow_fn = function_info.single_fn
                result = await workflow_fn("test input")
                assert result == "The workflow finished but no output was generated."

    @pytest.mark.asyncio
    async def test_autogen_team_workflow_exception(self, mock_config, mock_builder):
        """Test workflow exception handling."""
        with patch('nat_autogen_demo.register.AssistantAgent'), \
             patch('nat_autogen_demo.register.RoundRobinGroupChat') as mock_group_chat, \
             patch('nat_autogen_demo.register.StreamableHttpMcpToolAdapter'), \
             patch('nat_autogen_demo.register.StreamableHttpServerParams'), \
             patch('nat_autogen_demo.register.TextMentionTermination'), \
             patch('nat_autogen_demo.register.logger') as mock_logger:

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = []

            # Setup team to raise exception
            mock_team_instance = AsyncMock()
            mock_group_chat.return_value = mock_team_instance
            mock_team_instance.run.side_effect = Exception("Test error")

            from nat_autogen_demo.register import autogen_team

            async with autogen_team(mock_config, mock_builder) as function_info:
                workflow_fn = function_info.single_fn
                result = await workflow_fn("test input")
                assert "Error occurred during AutoGen workflow: Test error" in result
                mock_logger.exception.assert_called_with("Error in AutoGen workflow")

    @pytest.mark.asyncio
    async def test_autogen_team_builder_exception(self, mock_config, mock_builder):
        """Test exception handling during builder operations."""
        with patch('nat_autogen_demo.register.logger') as mock_logger:
            # Setup builder to raise exception
            mock_builder.get_llm.side_effect = Exception("Builder error")

            from nat_autogen_demo.register import autogen_team

            with pytest.raises(Exception, match="Builder error"):
                async with autogen_team(mock_config, mock_builder):
                    pass

            mock_logger.exception.assert_called_with("Failed to initialize AutoGen workflow")

    @pytest.mark.asyncio
    async def test_autogen_team_generator_exit(self, mock_config, mock_builder):
        """Test GeneratorExit handling."""
        with patch('nat_autogen_demo.register.logger') as mock_logger:
            from nat_autogen_demo.register import autogen_team

            # Create the context manager
            context_manager = autogen_team(mock_config, mock_builder)

            # Simulate GeneratorExit by closing the generator
            try:
                await context_manager.__aenter__()
                await context_manager.__aexit__(GeneratorExit, None, None)
            except GeneratorExit:
                pass

            mock_logger.info.assert_called_with("AutoGen workflow exited early")

    @pytest.mark.asyncio
    async def test_autogen_team_cleanup_logging(self, mock_config, mock_builder):
        """Test cleanup logging in finally block."""
        with patch('nat_autogen_demo.register.AssistantAgent'), \
             patch('nat_autogen_demo.register.RoundRobinGroupChat'), \
             patch('nat_autogen_demo.register.StreamableHttpMcpToolAdapter'), \
             patch('nat_autogen_demo.register.StreamableHttpServerParams'), \
             patch('nat_autogen_demo.register.TextMentionTermination'), \
             patch('nat_autogen_demo.register.logger') as mock_logger:

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = []

            from nat_autogen_demo.register import autogen_team

            async with autogen_team(mock_config, mock_builder):
                pass

            # Verify cleanup logging was called
            mock_logger.debug.assert_called_with("AutoGen workflow cleanup completed")

    @pytest.mark.asyncio
    async def test_autogen_team_agents_creation(self, mock_config, mock_builder):
        """Test that agents are created with correct parameters."""
        with patch('nat_autogen_demo.register.AssistantAgent') as mock_agent, \
             patch('nat_autogen_demo.register.RoundRobinGroupChat'), \
             patch('nat_autogen_demo.register.StreamableHttpMcpToolAdapter'), \
             patch('nat_autogen_demo.register.StreamableHttpServerParams'), \
             patch('nat_autogen_demo.register.TextMentionTermination'):

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = []

            from nat_autogen_demo.register import autogen_team

            async with autogen_team(mock_config, mock_builder):
                # Verify agents were created
                assert mock_agent.call_count == 2

                # Check agent configurations
                agent_calls = mock_agent.call_args_list

                # First agent (WeatherAndTimeAgent)
                weather_agent_call = agent_calls[0]
                assert weather_agent_call[1]['name'] == "WeatherAndTimeAgent"
                assert weather_agent_call[1]['model_client'] == mock_llm_client
                assert "weather and time information" in weather_agent_call[1]['system_message']
                assert "DONE" in weather_agent_call[1]['system_message']

                # Second agent (FinalResponseAgent)
                final_agent_call = agent_calls[1]
                assert final_agent_call[1]['name'] == "FinalResponseAgent"
                assert final_agent_call[1]['model_client'] == mock_llm_client
                assert "final response agent" in final_agent_call[1]['system_message']
                assert "APPROVE" in final_agent_call[1]['system_message']

    @pytest.mark.asyncio
    async def test_autogen_team_roundrobin_creation(self, mock_config, mock_builder):
        """Test RoundRobinGroupChat creation."""
        with patch('nat_autogen_demo.register.RoundRobinGroupChat') as mock_group_chat, \
             patch('nat_autogen_demo.register.StreamableHttpMcpToolAdapter'), \
             patch('nat_autogen_demo.register.StreamableHttpServerParams'), \
             patch('nat_autogen_demo.register.TextMentionTermination') as mock_termination:

            # Setup mocks
            mock_llm_client = AsyncMock()
            mock_builder.get_llm.return_value = mock_llm_client
            mock_builder.get_tools.return_value = []

            from nat_autogen_demo.register import autogen_team

            async with autogen_team(mock_config, mock_builder):
                # Verify RoundRobinGroupChat was created
                mock_group_chat.assert_called_once()

                # Check the call arguments
                call_args = mock_group_chat.call_args
                assert 'participants' in call_args[1]
                assert 'termination_condition' in call_args[1]
                assert len(call_args[1]['participants']) == 2

                # Verify termination condition
                mock_termination.assert_called_once_with("APPROVE")


class TestIntegration:
    """Integration tests for basic functionality."""

    def test_complete_config_creation(self):
        """Test complete configuration creation process."""
        # Create a complete configuration
        llm_ref = LLMRef("gpt-4")
        tools = ["weather_api", "time_api", "calculator"]

        config = AutoGenFunctionConfig(llm_name=llm_ref, tool_names=tools)

        # Verify all components
        assert config.llm_name == llm_ref
        assert config.tool_names == tools
        assert config.type == "autogen_team"

    def test_function_registration_structure(self):
        """Test that function registration is properly structured."""
        from nat_autogen_demo.register import autogen_team

        # The function should be callable (wrapped by decorator)
        assert callable(autogen_team)

        # Should have documentation
        assert autogen_team.__doc__ is not None


if __name__ == "__main__":
    pytest.main([__file__])

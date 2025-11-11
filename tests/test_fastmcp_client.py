# tests/test_fastmcp_client.py
"""
Unit tests for FastMCP Groq Client
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from client.fastmcp_groq_client import RobotFastMCPClient


class TestRobotFastMCPClient:
    """Test suite for RobotFastMCPClient."""

    @pytest.fixture
    def mock_groq_client(self):
        """Create a mock Groq client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response", tool_calls=None))]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock FastMCP client."""
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                MagicMock(name="test_tool", description="Test tool", inputSchema={"type": "object", "properties": {}})
            ]
        )
        return mock_client

    @pytest.fixture
    def client(self, mock_groq_client):
        """Create a RobotFastMCPClient instance with mocked dependencies."""
        with patch("client.fastmcp_groq_client.Groq", return_value=mock_groq_client):
            with patch("client.fastmcp_groq_client.Client") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client

                client = RobotFastMCPClient(groq_api_key="test_key", model="test-model")
                client.client = mock_client
                return client

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        with patch("client.fastmcp_groq_client.Groq"):
            with patch("client.fastmcp_groq_client.Client"):
                client = RobotFastMCPClient(groq_api_key="test_key", model="test-model")

                assert client.groq_api_key == "test_key"
                assert client.model == "test-model"
                assert client.available_tools == []
                assert client.conversation_history == []

    @pytest.mark.asyncio
    async def test_connect(self, client):
        """Test connecting to MCP server."""
        mock_tools = [
            MagicMock(name="tool1", description="Tool 1", inputSchema={}),
            MagicMock(name="tool2", description="Tool 2", inputSchema={}),
        ]
        client.client.list_tools = AsyncMock(return_value=mock_tools)
        client.client.__aenter__ = AsyncMock(return_value=client.client)

        await client.connect()

        assert len(client.available_tools) == 2
        client.client.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test disconnecting from MCP server."""
        client.client.__aexit__ = AsyncMock()

        await client.disconnect()

        client.client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_success(self, client):
        """Test successful tool call."""
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Tool executed successfully")]
        client.client.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("test_tool", {"arg1": "value1"})

        assert result == "Tool executed successfully"
        client.client.call_tool.assert_called_once_with("test_tool", {"arg1": "value1"})

    @pytest.mark.asyncio
    async def test_call_tool_error(self, client):
        """Test tool call with error."""
        client.client.call_tool = AsyncMock(side_effect=Exception("Tool error"))

        result = await client.call_tool("test_tool", {})

        assert "Error calling tool" in result
        assert "Tool error" in result

    @pytest.mark.asyncio
    async def test_chat_simple_response(self, client, mock_groq_client):
        """Test simple chat without tool calls."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Simple response", tool_calls=None))]
        client.groq_client.chat.completions.create.return_value = mock_response

        response = await client.chat("Hello")

        assert response == "Simple response"
        assert len(client.conversation_history) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self, client, mock_groq_client):
        """Test chat that requires tool calls."""
        # First response with tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock(message=MagicMock(content="", tool_calls=[mock_tool_call]))]

        # Second response after tool execution
        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock(message=MagicMock(content="Final response", tool_calls=None))]

        client.groq_client.chat.completions.create.side_effect = [mock_response1, mock_response2]

        # Mock tool execution
        client.client.call_tool = AsyncMock(return_value=MagicMock(content=[MagicMock(text="Tool result")]))

        response = await client.chat("Test command")

        assert response == "Final response"
        assert client.groq_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_convert_tools_to_groq_format(self, client):
        """Test tool format conversion."""
        client.available_tools = [
            MagicMock(name="test_tool", description="Test description", inputSchema={"type": "object", "properties": {}})
        ]

        groq_tools = client._convert_tools_to_groq_format()

        assert len(groq_tools) == 1
        assert groq_tools[0]["type"] == "function"
        assert groq_tools[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_process_tool_calls(self, client):
        """Test processing multiple tool calls."""
        mock_tool_call1 = MagicMock()
        mock_tool_call1.id = "call_1"
        mock_tool_call1.function.name = "tool1"
        mock_tool_call1.function.arguments = '{"arg": "value1"}'

        mock_tool_call2 = MagicMock()
        mock_tool_call2.id = "call_2"
        mock_tool_call2.function.name = "tool2"
        mock_tool_call2.function.arguments = '{"arg": "value2"}'

        client.client.call_tool = AsyncMock(return_value=MagicMock(content=[MagicMock(text="Result")]))

        results = await client.process_tool_calls([mock_tool_call1, mock_tool_call2])

        assert len(results) == 2
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert results[1]["tool_call_id"] == "call_2"

    @pytest.mark.asyncio
    async def test_max_iterations(self, client, mock_groq_client):
        """Test that chat stops after max iterations."""
        # Always return tool calls to trigger infinite loop
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = "{}"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="", tool_calls=[mock_tool_call]))]

        client.groq_client.chat.completions.create.return_value = mock_response
        client.client.call_tool = AsyncMock(return_value=MagicMock(content=[MagicMock(text="Result")]))

        response = await client.chat("Test")

        assert "Maximum iterations reached" in response

    def test_system_prompt(self, client):
        """Test that system prompt is properly set."""
        assert "robot control assistant" in client.system_prompt.lower()
        assert "coordinate system" in client.system_prompt.lower()
        assert "workspace" in client.system_prompt.lower()


class TestIntegrationScenarios:
    """Integration test scenarios."""

    @pytest.mark.asyncio
    async def test_complete_pick_and_place_flow(self):
        """Test complete pick-and-place workflow."""
        with patch("client.fastmcp_groq_client.Groq") as mock_groq:
            with patch("client.fastmcp_groq_client.Client") as mock_client_class:
                # Setup mocks
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()

                # Mock tools
                mock_tools = [
                    MagicMock(name="get_detected_objects", description="Get objects", inputSchema={}),
                    MagicMock(name="pick_place_object", description="Pick and place", inputSchema={}),
                ]
                mock_client.list_tools = AsyncMock(return_value=mock_tools)

                # Mock Groq responses
                mock_groq_instance = mock_groq.return_value

                # First call: get objects
                tool_call1 = MagicMock()
                tool_call1.id = "call_1"
                tool_call1.function.name = "get_detected_objects"
                tool_call1.function.arguments = "{}"

                response1 = MagicMock()
                response1.choices = [MagicMock(message=MagicMock(content="", tool_calls=[tool_call1]))]

                # Second call: pick and place
                tool_call2 = MagicMock()
                tool_call2.id = "call_2"
                tool_call2.function.name = "pick_place_object"
                tool_call2.function.arguments = json.dumps(
                    {
                        "object_name": "pencil",
                        "pick_coordinate": [0.15, -0.05],
                        "place_coordinate": [0.20, 0.10],
                        "location": "right next to",
                    }
                )

                response2 = MagicMock()
                response2.choices = [MagicMock(message=MagicMock(content="", tool_calls=[tool_call2]))]

                # Final response
                response3 = MagicMock()
                response3.choices = [
                    MagicMock(message=MagicMock(content="I've placed the pencil to the right of the cube.", tool_calls=None))
                ]

                mock_groq_instance.chat.completions.create.side_effect = [response1, response2, response3]

                # Mock tool results
                mock_client.call_tool = AsyncMock(
                    side_effect=[
                        MagicMock(content=[MagicMock(text='[{"label": "pencil", "x": 0.15, "y": -0.05}]')]),
                        MagicMock(content=[MagicMock(text="Success")]),
                    ]
                )

                # Execute
                client = RobotFastMCPClient("test_key")
                client.client = mock_client
                await client.connect()

                response = await client.chat("Pick up the pencil and place it next to the cube")

                assert "pencil" in response.lower()
                assert mock_client.call_tool.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

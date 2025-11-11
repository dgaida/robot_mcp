# tests/test_integration.py
"""
Integration tests for Robot MCP system
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_pick_and_place_workflow(self):
        """Test complete workflow from user input to robot action."""
        from client.fastmcp_groq_client import RobotFastMCPClient

        # Mock all external dependencies
        with patch("client.fastmcp_groq_client.Groq") as mock_groq:
            with patch("client.fastmcp_groq_client.Client") as mock_client_class:
                # Setup MCP client mock
                mock_mcp_client = AsyncMock()
                mock_client_class.return_value = mock_mcp_client
                mock_mcp_client.__aenter__ = AsyncMock(return_value=mock_mcp_client)
                mock_mcp_client.__aexit__ = AsyncMock()

                # Mock available tools
                mock_tools = [
                    MagicMock(
                        name="get_detected_objects",
                        description="Get detected objects",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    MagicMock(
                        name="pick_place_object",
                        description="Pick and place object",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
                mock_mcp_client.list_tools = AsyncMock(return_value=mock_tools)

                # Setup Groq mock
                mock_groq_instance = mock_groq.return_value

                # Simulate tool call flow
                import json

                # Step 1: LLM calls get_detected_objects
                tool_call_1 = MagicMock()
                tool_call_1.id = "call_1"
                tool_call_1.function.name = "get_detected_objects"
                tool_call_1.function.arguments = "{}"

                response_1 = MagicMock()
                response_1.choices = [MagicMock(message=MagicMock(content="", tool_calls=[tool_call_1]))]

                # Step 2: LLM calls pick_place_object
                tool_call_2 = MagicMock()
                tool_call_2.id = "call_2"
                tool_call_2.function.name = "pick_place_object"
                tool_call_2.function.arguments = json.dumps(
                    {
                        "object_name": "pencil",
                        "pick_coordinate": [0.15, -0.05],
                        "place_coordinate": [0.20, 0.10],
                        "location": "right next to",
                    }
                )

                response_2 = MagicMock()
                response_2.choices = [MagicMock(message=MagicMock(content="", tool_calls=[tool_call_2]))]

                # Step 3: Final response
                response_3 = MagicMock()
                response_3.choices = [
                    MagicMock(
                        message=MagicMock(
                            content="I have successfully picked up the pencil and placed it to the right of the cube.",
                            tool_calls=None,
                        )
                    )
                ]

                mock_groq_instance.chat.completions.create.side_effect = [response_1, response_2, response_3]

                # Mock tool execution results
                objects_result = MagicMock()
                objects_result.content = [
                    MagicMock(
                        text=json.dumps([{"label": "pencil", "x": 0.15, "y": -0.05}, {"label": "cube", "x": 0.20, "y": 0.10}])
                    )
                ]

                pick_place_result = MagicMock()
                pick_place_result.content = [MagicMock(text="Successfully picked and placed object")]

                mock_mcp_client.call_tool = AsyncMock(side_effect=[objects_result, pick_place_result])

                # Execute workflow
                client = RobotFastMCPClient(groq_api_key="test_key")
                client.client = mock_mcp_client
                await client.connect()

                response = await client.chat("Pick up the pencil and place it to the right of the cube")

                # Verify workflow
                assert "successfully" in response.lower() or "pencil" in response.lower()
                assert mock_mcp_client.call_tool.call_count == 2
                assert mock_groq_instance.chat.completions.create.call_count == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_object_detection_and_sorting(self):
        """Test object detection followed by sorting."""
        from client.fastmcp_groq_client import RobotFastMCPClient

        with patch("client.fastmcp_groq_client.Groq") as mock_groq:
            with patch("client.fastmcp_groq_client.Client") as mock_client_class:
                mock_mcp_client = AsyncMock()
                mock_client_class.return_value = mock_mcp_client
                mock_mcp_client.__aenter__ = AsyncMock(return_value=mock_mcp_client)

                mock_tools = [
                    MagicMock(name="get_detected_objects_sorted", description="", inputSchema={}),
                ]
                mock_mcp_client.list_tools = AsyncMock(return_value=mock_tools)

                mock_groq_instance = mock_groq.return_value

                # Tool call
                import json

                tool_call = MagicMock()
                tool_call.id = "call_sort"
                tool_call.function.name = "get_detected_objects_sorted"
                tool_call.function.arguments = '{"ascending": true}'

                response_1 = MagicMock()
                response_1.choices = [MagicMock(message=MagicMock(content="", tool_calls=[tool_call]))]

                response_2 = MagicMock()
                response_2.choices = [
                    MagicMock(
                        message=MagicMock(
                            content="Objects sorted from smallest to largest: pencil, cube, box", tool_calls=None
                        )
                    )
                ]

                mock_groq_instance.chat.completions.create.side_effect = [response_1, response_2]

                sorted_result = MagicMock()
                sorted_result.content = [
                    MagicMock(
                        text=json.dumps(
                            [
                                {"label": "pencil", "area": 0.0018},
                                {"label": "cube", "area": 0.0025},
                                {"label": "box", "area": 0.0100},
                            ]
                        )
                    )
                ]
                mock_mcp_client.call_tool = AsyncMock(return_value=sorted_result)

                client = RobotFastMCPClient(groq_api_key="test_key")
                client.client = mock_mcp_client
                await client.connect()

                response = await client.chat("Sort all objects by size")

                assert "smallest" in response.lower() or "sorted" in response.lower()


class TestServerClientCommunication:
    """Test MCP server-client communication."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tool_list_synchronization(self):
        """Test that client receives all server tools."""
        from client.fastmcp_groq_client import RobotFastMCPClient

        with patch("client.fastmcp_groq_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)

            # Simulate server tools
            expected_tools = [
                MagicMock(name="pick_place_object", description="", inputSchema={}),
                MagicMock(name="get_detected_objects", description="", inputSchema={}),
                MagicMock(name="speak", description="", inputSchema={}),
            ]
            mock_client.list_tools = AsyncMock(return_value=expected_tools)

            with patch("client.fastmcp_groq_client.Groq"):
                client = RobotFastMCPClient(groq_api_key="test_key")
                client.client = mock_client
                await client.connect()

            assert len(client.available_tools) == 3
            tool_names = [t["function"]["name"] for t in client.available_tools]
            assert "pick_place_object" in tool_names
            assert "get_detected_objects" in tool_names
            assert "speak" in tool_names

    @pytest.mark.integration
    def test_tool_parameter_validation(self):
        """Test that tool parameters are properly validated."""
        from server.fastmcp_robot_server import pick_place_object

        with patch("server.fastmcp_robot_server.robot") as mock_robot:
            mock_robot.pick_place_object.return_value = True

            # Valid call
            result = pick_place_object(
                object_name="pencil", pick_coordinate=[0.15, -0.05], place_coordinate=[0.20, 0.10], location="right next to"
            )
            assert result is True

            # Verify parameters passed correctly
            call_args = mock_robot.pick_place_object.call_args
            assert call_args.kwargs["object_name"] == "pencil"
            assert call_args.kwargs["pick_coordinate"] == [0.15, -0.05]


class TestErrorRecovery:
    """Test error handling and recovery."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors."""
        from client.fastmcp_groq_client import RobotFastMCPClient

        with patch("client.fastmcp_groq_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.list_tools = AsyncMock(return_value=[])

            # Simulate tool execution error
            mock_client.call_tool = AsyncMock(side_effect=Exception("Robot communication error"))

            with patch("client.fastmcp_groq_client.Groq"):
                client = RobotFastMCPClient(groq_api_key="test_key")
                client.client = mock_client
                await client.connect()

                result = await client.call_tool("test_tool", {})

                assert "Error" in result
                assert "Robot communication error" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self):
        """Test recovery from connection failures."""
        from client.fastmcp_groq_client import RobotFastMCPClient

        with patch("client.fastmcp_groq_client.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # First connection attempt fails
            mock_client.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))

            with patch("client.fastmcp_groq_client.Groq"):
                client = RobotFastMCPClient(groq_api_key="test_key")
                client.client = mock_client

                with pytest.raises(Exception, match="Connection failed"):
                    await client.connect()


class TestConcurrentOperations:
    """Test concurrent operations."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_concurrent_commands(self):
        """Test handling multiple concurrent commands."""
        from client.fastmcp_groq_client import RobotFastMCPClient

        with patch("client.fastmcp_groq_client.Groq") as mock_groq:
            with patch("client.fastmcp_groq_client.Client") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.list_tools = AsyncMock(return_value=[])

                mock_groq_instance = mock_groq.return_value
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="Done", tool_calls=None))]
                mock_groq_instance.chat.completions.create.return_value = mock_response

                client = RobotFastMCPClient(groq_api_key="test_key")
                client.client = mock_client
                await client.connect()

                # Execute multiple commands concurrently
                tasks = [client.chat("Command 1"), client.chat("Command 2"), client.chat("Command 3")]

                results = await asyncio.gather(*tasks)

                assert len(results) == 3
                assert all(isinstance(r, str) for r in results)


class TestDataFlowIntegrity:
    """Test data flow integrity through the system."""

    @pytest.mark.integration
    def test_coordinate_transformation_consistency(self):
        """Test coordinate transformations are consistent."""
        # Test coordinates in valid range
        test_coords = [[0.25, 0.0], [0.337, 0.087], [0.163, -0.087]]  # Center  # Upper left  # Lower right

        for coord in test_coords:
            assert len(coord) == 2
            assert isinstance(coord[0], (int, float))
            assert isinstance(coord[1], (int, float))
            # Niryo workspace bounds
            assert 0.163 <= coord[0] <= 0.337
            assert -0.087 <= coord[1] <= 0.087

    @pytest.mark.integration
    def test_object_data_serialization(self):
        """Test object data can be serialized/deserialized."""
        import json

        test_object = {"label": "pencil", "x": 0.15, "y": -0.05, "width_m": 0.015, "height_m": 0.120, "area_m2": 0.0018}

        # Serialize
        json_str = json.dumps(test_object)

        # Deserialize
        recovered = json.loads(json_str)

        assert recovered["label"] == "pencil"
        assert recovered["x"] == 0.15
        assert recovered["area_m2"] == 0.0018


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

# tests/test_gui.py
"""
Unit tests for Gradio GUI (RobotMCPGUI)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from robot_gui.mcp_app import RobotMCPGUI, create_gradio_interface


class TestRobotMCPGUI:
    """Test suite for RobotMCPGUI."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mocked dependencies."""
        with patch("robot_gui.mcp_app.RobotUniversalMCPClient") as mock_client:
            with patch("robot_gui.mcp_app.Speech2Text") as mock_speech:
                with patch("robot_gui.mcp_app.RedisImageStreamer") as mock_streamer:
                    with patch("robot_gui.mcp_app.RedisTextOverlayManager") as mock_text:
                        yield {
                            "client": mock_client,
                            "speech": mock_speech,
                            "streamer": mock_streamer,
                            "text": mock_text
                        }

    @pytest.fixture
    def gui(self, mock_dependencies):
        """Create GUI instance with mocked dependencies."""
        gui = RobotMCPGUI(
            api_choice="groq",
            model="test-model",
            robot_id="niryo",
            use_simulation=True,
            redis_host="localhost",
            redis_port=6379
        )
        return gui

    def test_initialization(self, gui):
        """Test GUI initialization."""
        assert gui.api_choice == "groq"
        assert gui.model == "test-model"
        assert gui.robot_id == "niryo"
        assert gui.use_simulation is True
        assert gui.mcp_connected is False
        assert gui.chat_history == []

    @pytest.mark.asyncio
    async def test_connect_mcp(self, gui, mock_dependencies):
        """Test connecting MCP client."""
        mock_client_instance = AsyncMock()
        mock_client_instance.connect = AsyncMock()

        tool1 = MagicMock()
        tool1.name = "tool1"
        tool2 = MagicMock()
        tool2.name = "tool2"
        mock_client_instance.available_tools = [tool1, tool2]

        mock_dependencies["client"].return_value = mock_client_instance

        # Mock HAS_MCP_CLIENT to True
        with patch("robot_gui.mcp_app.HAS_MCP_CLIENT", True):
            success, message = await gui.connect_mcp()

        assert success is True
        assert "Connected" in message
        assert gui.mcp_connected is True
        assert gui.mcp_client is not None

    @pytest.mark.asyncio
    async def test_connect_mcp_failure(self, gui, mock_dependencies):
        """Test MCP client connection failure."""
        mock_client_instance = AsyncMock()
        mock_client_instance.connect = AsyncMock(side_effect=Exception("Connection error"))
        mock_dependencies["client"].return_value = mock_client_instance

        with patch("robot_gui.mcp_app.HAS_MCP_CLIENT", True):
            success, message = await gui.connect_mcp()

        assert success is False
        assert "Failed to connect" in message
        assert gui.mcp_connected is False

    @pytest.mark.asyncio
    async def test_process_chat_not_connected(self, gui):
        """Test processing chat when not connected."""
        gui.mcp_connected = False
        history = []

        async for updated_history in gui.process_chat("Hello", history):
            pass

        assert len(updated_history) == 1
        assert "not connected" in updated_history[0][1].lower()

    @pytest.mark.asyncio
    async def test_process_chat_success(self, gui, mock_dependencies):
        """Test successful chat processing."""
        gui.mcp_connected = True
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value="Robot response")
        gui.mcp_client = mock_client

        history = []
        updates = []
        async for updated_history in gui.process_chat("Pick up pencil", history):
            updates.append(list(updated_history))

        assert len(updates) > 0
        final_history = updates[-1]
        assert final_history[0] == ("Pick up pencil", "Robot response")
        mock_client.chat.assert_called_once_with("Pick up pencil")

    def test_record_voice_success(self, gui, mock_dependencies):
        """Test successful voice recording."""
        mock_speech = MagicMock()
        mock_speech.record_and_transcribe.return_value = "Voice command"
        gui.speech2text = mock_speech

        result = gui.record_voice()
        assert result == "Voice command"

    def test_record_voice_not_initialized(self, gui):
        """Test voice recording when not initialized."""
        gui.speech2text = None
        result = gui.record_voice()
        assert "not initialized" in result.lower()

    def test_get_latest_frame_available(self, gui, mock_dependencies):
        """Test getting latest frame when available."""
        mock_streamer = MagicMock()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_streamer.get_latest_image.return_value = (test_image, {})
        gui.image_streamer = mock_streamer

        frame = gui.get_latest_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_get_status_html(self, gui):
        """Test status HTML generation."""
        gui.mcp_connected = True
        html = gui.get_status_html()
        assert "Connected" in html
        assert "Simulation" in html
        assert gui.robot_id.upper() in html


class TestGradioInterface:
    """Test Gradio interface creation."""

    def test_create_interface(self):
        """Test Gradio interface creation."""
        gui = MagicMock(spec=RobotMCPGUI)
        gui.get_status_html.return_value = "<div>Status</div>"

        with patch("robot_gui.mcp_app.gr.Blocks") as mock_blocks, \
             patch("robot_gui.mcp_app.gr.Row"), \
             patch("robot_gui.mcp_app.gr.Column"), \
             patch("robot_gui.mcp_app.gr.HTML"), \
             patch("robot_gui.mcp_app.gr.Markdown"), \
             patch("robot_gui.mcp_app.gr.Textbox"), \
             patch("robot_gui.mcp_app.gr.Button"), \
             patch("robot_gui.mcp_app.gr.Chatbot"), \
             patch("robot_gui.mcp_app.gr.Examples"), \
             patch("robot_gui.mcp_app.gr.Image"), \
             patch("robot_gui.mcp_app.gr.Timer"):

            mock_demo = MagicMock()
            mock_blocks.return_value.__enter__.return_value = mock_demo

            demo = create_gradio_interface(gui)

        assert demo is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

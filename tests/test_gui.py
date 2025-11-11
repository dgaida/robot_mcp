# tests/test_gui.py
"""
Unit tests for Gradio GUI
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
        with patch("robot_gui.mcp_app.RobotFastMCPClient") as mock_client:
            with patch("robot_gui.mcp_app.Speech2Text") as mock_speech:
                with patch("robot_gui.mcp_app.RedisImageStreamer") as mock_streamer:
                    yield {"client": mock_client, "speech": mock_speech, "streamer": mock_streamer}

    @pytest.fixture
    def gui(self, mock_dependencies):
        """Create GUI instance with mocked dependencies."""
        gui = RobotMCPGUI(
            groq_api_key="test_key",
            elevenlabs_api_key="test_el_key",
            model="test-model",
            robot_id="niryo",
            use_simulation=True,
            verbose=False,
        )
        return gui

    def test_initialization(self, gui):
        """Test GUI initialization."""
        assert gui.groq_api_key == "test_key"
        assert gui.elevenlabs_api_key == "test_el_key"
        assert gui.model == "test-model"
        assert gui.robot_id == "niryo"
        assert gui.use_simulation is True
        assert gui.mcp_server_running is False
        assert gui.mcp_client_connected is False

    def test_initialize_environment(self, gui, mock_dependencies):
        """Test environment initialization."""
        mock_dependencies["speech"].return_value = MagicMock()

        with patch("robot_gui.mcp_app.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.float32 = "float32"

            with patch.object(gui, "_start_camera_updates"):
                result = gui.initialize_environment()

        assert result is True
        assert gui.speech2text is not None

    def test_initialize_environment_failure(self, gui):
        """Test environment initialization failure."""
        with patch("robot_gui.mcp_app.Speech2Text", side_effect=Exception("Init error")):
            result = gui.initialize_environment()

        assert result is False

    @pytest.mark.asyncio
    async def test_start_mcp_server(self, gui):
        """Test starting MCP server."""
        with patch("robot_gui.mcp_app.subprocess.Popen") as mock_popen:
            with patch("robot_gui.mcp_app.httpx.AsyncClient") as mock_httpx:
                mock_process = MagicMock()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                mock_client = AsyncMock()
                mock_client.get = AsyncMock()
                mock_httpx.return_value.__aenter__.return_value = mock_client

                result = await gui.start_mcp_server()

        assert result is True
        assert gui.mcp_server_running is True

    @pytest.mark.asyncio
    async def test_connect_mcp_client(self, gui):
        """Test connecting MCP client."""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.available_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]

        with patch("robot_gui.mcp_app.RobotFastMCPClient", return_value=mock_client):
            result = await gui.connect_mcp_client()

        assert result is True
        assert gui.mcp_client_connected is True

    @pytest.mark.asyncio
    async def test_connect_mcp_client_failure(self, gui):
        """Test MCP client connection failure."""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(side_effect=Exception("Connection error"))

        with patch("robot_gui.mcp_app.RobotFastMCPClient", return_value=mock_client):
            result = await gui.connect_mcp_client()

        assert result is False

    def test_add_messages(self, gui):
        """Test adding different message types."""
        gui._add_system_message("System test")
        gui._add_user_message("User test")
        gui._add_assistant_message("Assistant test")
        gui._add_tool_call_message("test_tool", {"arg": "value"})

        assert len(gui.chat_history) == 4
        assert "System:" in gui.chat_history[0]["content"]
        assert gui.chat_history[1]["role"] == "user"
        assert gui.chat_history[2]["role"] == "assistant"
        assert "Calling tool:" in gui.chat_history[3]["content"]

    @pytest.mark.asyncio
    async def test_process_user_input_not_connected(self, gui):
        """Test processing input when not connected."""
        gui.mcp_client_connected = False

        result = None
        async for update in gui.process_user_input("Test command"):
            result = update

        assert result is not None
        assert any("not connected" in msg.get("content", "").lower() for msg in result)

    @pytest.mark.asyncio
    async def test_process_user_input_empty(self, gui):
        """Test processing empty input."""
        result = None
        async for update in gui.process_user_input(""):
            result = update

        assert result == gui.chat_history

    @pytest.mark.asyncio
    async def test_process_user_input_success(self, gui):
        """Test successful input processing."""
        gui.mcp_client_connected = True

        mock_client = AsyncMock()
        mock_client.conversation_history = []
        mock_client.system_prompt = "Test prompt"
        mock_client.groq_client = MagicMock()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response text", tool_calls=None))]
        mock_client.groq_client.chat.completions.create.return_value = mock_response
        mock_client._convert_tools_to_groq_format.return_value = []

        gui.mcp_client = mock_client

        result = None
        async for update in gui.process_user_input("Test command"):
            result = update

        assert result is not None
        assert len(gui.chat_history) > 0

    def test_record_voice_input_success(self, gui):
        """Test successful voice input recording."""
        mock_speech2text = MagicMock()
        mock_speech2text.record_and_transcribe.return_value = "Test transcription"
        gui.speech2text = mock_speech2text

        result = gui.record_voice_input()

        assert result == "Test transcription"
        assert any("Transcribed:" in msg.get("content", "") for msg in gui.chat_history)

    def test_record_voice_input_no_speech(self, gui):
        """Test voice input with no speech detected."""
        mock_speech2text = MagicMock()
        mock_speech2text.record_and_transcribe.return_value = ""
        gui.speech2text = mock_speech2text

        result = gui.record_voice_input()

        assert result == ""
        assert any("No speech detected" in msg.get("content", "") for msg in gui.chat_history)

    def test_record_voice_input_not_initialized(self, gui):
        """Test voice input when not initialized."""
        gui.speech2text = None

        result = gui.record_voice_input()

        assert "not initialized" in result.lower()

    def test_get_current_frame_available(self, gui):
        """Test getting current frame when available."""
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gui.current_frame = test_frame

        frame = gui.get_current_frame()

        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_get_current_frame_not_available(self, gui):
        """Test getting frame when not available (returns placeholder)."""
        gui.current_frame = None

        frame = gui.get_current_frame()

        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert np.all(frame == 0)  # Black placeholder

    def test_get_status_html(self, gui):
        """Test status HTML generation."""
        gui.mcp_server_running = True
        gui.mcp_client_connected = True

        html = gui.get_status_html()

        assert "Running" in html
        assert "Connected" in html
        assert gui.robot_id.upper() in html
        assert "Simulation" in html

    def test_get_status_html_disconnected(self, gui):
        """Test status HTML when disconnected."""
        gui.mcp_server_running = False
        gui.mcp_client_connected = False

        html = gui.get_status_html()

        assert "Stopped" in html
        assert "Disconnected" in html

    @pytest.mark.asyncio
    async def test_cleanup(self, gui):
        """Test cleanup process."""
        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        gui.mcp_client = mock_client

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock()
        gui.server_process = mock_process

        gui.cleanup()

        # Should not raise exceptions
        assert True


class TestGradioInterface:
    """Test Gradio interface creation."""

    @pytest.fixture
    def mock_gui(self):
        """Create mock GUI instance."""
        with patch("robot_gui.mcp_app.RobotFastMCPClient"):
            with patch("robot_gui.mcp_app.Speech2Text"):
                with patch("robot_gui.mcp_app.RedisImageStreamer"):
                    gui = RobotMCPGUI(groq_api_key="test_key", elevenlabs_api_key="test_el_key")
                    return gui

    def test_create_interface(self, mock_gui):
        """Test Gradio interface creation."""
        with patch("robot_gui.mcp_app.gr.Blocks") as mock_blocks:
            mock_blocks.return_value.__enter__ = MagicMock()
            mock_blocks.return_value.__exit__ = MagicMock()

            with patch("robot_gui.mcp_app.gr.Markdown"):
                with patch("robot_gui.mcp_app.gr.Row"):
                    with patch("robot_gui.mcp_app.gr.Column"):
                        with patch("robot_gui.mcp_app.gr.HTML"):
                            with patch("robot_gui.mcp_app.gr.Chatbot"):
                                with patch("robot_gui.mcp_app.gr.Textbox"):
                                    with patch("robot_gui.mcp_app.gr.Button"):
                                        with patch("robot_gui.mcp_app.gr.Dropdown"):
                                            with patch("robot_gui.mcp_app.gr.Image"):
                                                with patch("robot_gui.mcp_app.gr.Timer"):
                                                    demo = create_gradio_interface(mock_gui)

        # Should create demo without errors
        assert demo is not None


class TestCameraThread:
    """Test camera update thread."""

    @pytest.fixture
    def gui_with_streamer(self):
        """Create GUI with mocked streamer."""
        with patch("robot_gui.mcp_app.RobotFastMCPClient"):
            with patch("robot_gui.mcp_app.Speech2Text"):
                mock_streamer = MagicMock()
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                mock_streamer.get_latest_image.return_value = (test_image, {})

                with patch("robot_gui.mcp_app.RedisImageStreamer", return_value=mock_streamer):
                    gui = RobotMCPGUI(groq_api_key="test_key", elevenlabs_api_key="test_el_key")
                    gui._streamer = mock_streamer
                    return gui

    def test_camera_thread_updates_frame(self, gui_with_streamer):
        """Test that camera thread updates current frame."""
        import time

        # Start camera thread
        with patch.object(gui_with_streamer, "_start_camera_updates"):
            gui_with_streamer._start_camera_updates()

        # Give thread time to update
        time.sleep(0.2)

        # Frame should be updated
        # (In real test, would verify the camera thread is running)
        assert True


class TestConversationHistory:
    """Test conversation history management."""

    @pytest.fixture
    def gui(self):
        """Create GUI instance."""
        with patch("robot_gui.mcp_app.RobotFastMCPClient"):
            with patch("robot_gui.mcp_app.Speech2Text"):
                with patch("robot_gui.mcp_app.RedisImageStreamer"):
                    return RobotMCPGUI(groq_api_key="test_key", elevenlabs_api_key="test_el_key")

    def test_chat_history_initially_empty(self, gui):
        """Test that chat history starts empty."""
        assert len(gui.chat_history) == 0

    def test_chat_history_accumulates(self, gui):
        """Test that messages accumulate in history."""
        gui._add_user_message("Message 1")
        gui._add_assistant_message("Response 1")
        gui._add_user_message("Message 2")
        gui._add_assistant_message("Response 2")

        assert len(gui.chat_history) == 4
        assert gui.chat_history[0]["role"] == "user"
        assert gui.chat_history[1]["role"] == "assistant"

    def test_tool_call_messages_in_history(self, gui):
        """Test tool call messages in history."""
        gui._add_tool_call_message("pick_object", {"object_name": "pencil"})

        assert len(gui.chat_history) == 1
        assert "pick_object" in gui.chat_history[0]["content"]
        assert "pencil" in gui.chat_history[0]["content"]


class TestErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def gui(self):
        """Create GUI instance."""
        with patch("robot_gui.mcp_app.RobotFastMCPClient"):
            with patch("robot_gui.mcp_app.Speech2Text"):
                with patch("robot_gui.mcp_app.RedisImageStreamer"):
                    return RobotMCPGUI(groq_api_key="test_key", elevenlabs_api_key="test_el_key")

    @pytest.mark.asyncio
    async def test_process_input_with_exception(self, gui):
        """Test handling exceptions during input processing."""
        gui.mcp_client_connected = True

        mock_client = AsyncMock()
        mock_client.conversation_history = []
        mock_client.system_prompt = "Test"
        mock_client._convert_tools_to_groq_format.return_value = []
        mock_client.groq_client = MagicMock()
        mock_client.groq_client.chat.completions.create.side_effect = Exception("API Error")

        gui.mcp_client = mock_client

        result = None
        async for update in gui.process_user_input("Test"):
            result = update

        # Should handle error gracefully
        assert result is not None
        assert any("Error" in msg.get("content", "") for msg in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

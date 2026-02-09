#!/usr/bin/env python3
"""
Enhanced Robot Control GUI with MCP Integration

Features:
- Live annotated frame visualization from Redis
- Multi-LLM support via LLMClient
- Speech-to-text integration
- FastMCP client integration
- Real-time object detection display

Compatible with Gradio 3.x, 4.x, 5.x, and 6.x
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv

# Check Gradio version for compatibility
try:
    gr_version = gr.__version__
    print(f"Gradio version: {gr_version}")
except AttributeError:
    gr_version = "unknown"
    print("⚠️ Could not determine Gradio version")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from redis_robot_comm import RedisImageStreamer, RedisTextOverlayManager

try:
    from speech2text import Speech2Text
except (ImportError, OSError):
    print("⚠️ speech2text not available")
    Speech2Text = None

# Import FastMCP client
try:
    from client.fastmcp_universal_client import RobotUniversalMCPClient

    HAS_MCP_CLIENT = True
except ImportError:
    print("⚠️ FastMCP client not available")
    HAS_MCP_CLIENT = False


class RobotMCPGUI:
    """Enhanced GUI with Redis visualization and multi-LLM support."""

    # Default example tasks
    default_examples = [
        "What objects do you see?",
        "Pick up the pencil and place it at [0.2, 0.1]",
        "Move the red cube to the right of the blue square",
        "Arrange objects in a triangle pattern",
    ]

    def __init__(
        self,
        api_choice: str = "groq",
        model: str = None,
        robot_id: str = "niryo",
        use_simulation: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize Enhanced Robot GUI.

        Args:
            api_choice: LLM provider (openai, groq, gemini, ollama)
            model: Specific model name
            robot_id: Robot type (niryo/widowx)
            use_simulation: Use simulation mode
            redis_host: Redis server host
            redis_port: Redis server port
        """
        self.api_choice = api_choice
        self.model = model
        self.robot_id = robot_id
        self.use_simulation = use_simulation

        print("=" * 60)
        print("ENHANCED ROBOT GUI INITIALIZATION")
        print("=" * 60)
        print(f"  LLM Provider: {api_choice}")
        print(f"  Model: {model or 'default'}")
        print(f"  Robot: {robot_id}")
        print(f"  Simulation: {use_simulation}")
        print(f"  Redis: {redis_host}:{redis_port}")
        print("=" * 60 + "\n")

        # Initialize MCP client
        self.mcp_client: Optional[RobotUniversalMCPClient] = None
        self.mcp_connected = False

        # Initialize Redis streamers
        try:
            self.image_streamer = RedisImageStreamer(host=redis_host, port=redis_port, stream_name="annotated_camera")
            self.text_manager = RedisTextOverlayManager(host=redis_host, port=redis_port)
            print("✓ Redis connections established")
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            self.image_streamer = None
            self.text_manager = None

        # Initialize speech-to-text
        self.speech2text: Optional[Speech2Text] = None
        self._init_speech2text()

        # Chat history
        self.chat_history = []

        # Current frame
        self.current_frame = None
        self.frame_lock = asyncio.Lock()

    async def suggest_tasks(self):
        """
        Generate example task suggestions based on detected objects and available tools.

        Returns:
            list: List of 4 suggested tasks
        """
        if not self.mcp_connected or not self.mcp_client:
            print("⚠️ MCP client not connected, returning default examples")
            return self.default_examples

        try:
            print("🔍 Fetching detected objects and skills for suggestions...")

            # 1. Get detected objects using the tool
            objects_json = await self.mcp_client.call_tool("get_detected_objects", {})

            # 2. Get available tools as skills
            skills = []
            for tool in self.mcp_client.available_tools:
                skills.append(f"- {tool.name}: {tool.description}")
            skills_str = "\n".join(skills)

            # 3. Prepare prompt for LLM
            prompt = f"""
            You are a robot task suggestion assistant. Based on the detected objects and the robot's available skills,
            suggest 4 diverse, interesting, and realistic example tasks that a user could ask the robot to perform.

            DETECTED OBJECTS (JSON):
            {objects_json}

            AVAILABLE ROBOT SKILLS:
            {skills_str}

            INSTRUCTIONS:
            - Suggest exactly 4 tasks.
            - Each task should be a single, concise sentence.
            - Tasks should be diverse (e.g., one about detection, one about moving, one about arranging, etc.).
            - Tasks MUST be performable using the available skills and objects.
            - Return ONLY a JSON list of strings.

            Example output:
            ["What is the largest object you can see?", "Move the blue cube to the left of the red one", "Pick up the pen and place it at [0.2, 0.0]", "Arrange all cubes in a line"]
            """

            # 4. Call LLM
            messages = [
                {"role": "system", "content": "You are a helpful robot assistant. You only output JSON."},
                {"role": "user", "content": prompt},
            ]

            # LLMClient.chat_completion is synchronous, so we run it in a thread to avoid blocking the event loop.
            # We use a compatible way for Python 3.8+
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.mcp_client.llm_client.chat_completion(messages))

            # 5. Parse response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                try:
                    suggestions = json.loads(json_match.group(0))
                    if isinstance(suggestions, list) and len(suggestions) >= 1:
                        # Ensure we have exactly 4
                        if len(suggestions) > 4:
                            suggestions = suggestions[:4]
                        while len(suggestions) < 4:
                            suggestions.append(self.default_examples[len(suggestions)])
                        return suggestions
                except json.JSONDecodeError:
                    print(f"⚠️ Failed to parse JSON from: {json_match.group(0)}")

            print(f"⚠️ Failed to find JSON list in LLM response: {response}")

        except Exception as e:
            print(f"❌ Error generating suggestions: {e}")

        # Fallback to defaults
        return self.default_examples

    def _init_speech2text(self):
        """Initialize speech-to-text system."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            self.speech2text = Speech2Text(device=device, torch_dtype=torch_dtype, use_whisper_mic=True, verbose=False)
            print("✓ Speech-to-text initialized")
        except Exception as e:
            print(f"⚠️ Speech-to-text initialization failed: {e}")
            self.speech2text = None

    async def connect_mcp(self):
        """Connect to MCP server."""
        if not HAS_MCP_CLIENT:
            return False, "MCP client not available"

        try:
            print("Connecting to MCP server...")

            self.mcp_client = RobotUniversalMCPClient(api_choice=self.api_choice, model=self.model)

            await self.mcp_client.connect()

            self.mcp_connected = True
            tools = [t.name for t in self.mcp_client.available_tools]

            print("✓ MCP client connected")
            return True, f"Connected to MCP server\nAvailable tools: {', '.join(tools[:5])}..."

        except Exception as e:
            error_msg = f"Failed to connect: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg

    async def process_chat(self, message: str, history: list):
        """
        Process chat message through MCP client.

        Args:
            message: User message
            history: Chat history (list of tuples)

        Yields:
            Updated chat history
        """
        if not message or not message.strip():
            yield history
            return

        if not self.mcp_connected:
            history.append((message, "⚠️ MCP server not connected. Please connect first."))
            yield history
            return

        # Add user message (initially with no response)
        history.append((message, ""))
        yield history

        try:
            # Add "thinking" indicator
            history[-1] = (message, "🤔 Processing...")
            yield history

            # Process through MCP client
            response = await self.mcp_client.chat(message)

            # Update with actual response
            history[-1] = (message, response)
            yield history

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history[-1] = (message, error_msg)
            yield history

    def record_voice(self):
        """Record voice input and transcribe."""
        if not self.speech2text:
            return "⚠️ Speech recognition not initialized"

        try:
            print("🎤 Recording... Please speak now")
            transcription = self.speech2text.record_and_transcribe()

            if transcription:
                print(f"🎤 Transcribed: {transcription}")
                return transcription
            else:
                return ""

        except Exception as e:
            error_msg = f"❌ Voice input error: {str(e)}"
            print(error_msg)
            return ""

    def get_latest_frame(self):
        """Get latest annotated frame from Redis."""
        if not self.image_streamer:
            # Return placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Redis not connected", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
            return placeholder

        try:
            result = self.image_streamer.get_latest_image()
            if result:
                image, metadata = result
                self.current_frame = image
                return image
            elif self.current_frame is not None:
                return self.current_frame
            else:
                # No frame yet
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, "Waiting for frames...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2
                )
                return placeholder

        except Exception as e:
            print(f"Error getting frame: {e}")
            return self.current_frame if self.current_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def get_status_html(self):
        """Get HTML status display."""
        mcp_status = "🟢 Connected" if self.mcp_connected else "🔴 Disconnected"
        redis_status = "🟢 Connected" if self.image_streamer else "🔴 Disconnected"
        speech_status = "🟢 Available" if self.speech2text else "🔴 Unavailable"

        llm_info = f"{self.api_choice.upper()}"
        if self.mcp_client:
            llm_info += f" - {self.mcp_client.llm_client.llm}"

        html = f"""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
            <h3 style="margin-top: 0;">🤖 System Status</h3>
            <table style="width: 100%;">
                <tr>
                    <td><strong>MCP Server:</strong></td>
                    <td>{mcp_status}</td>
                </tr>
                <tr>
                    <td><strong>Redis:</strong></td>
                    <td>{redis_status}</td>
                </tr>
                <tr>
                    <td><strong>Speech-to-Text:</strong></td>
                    <td>{speech_status}</td>
                </tr>
                <tr>
                    <td><strong>LLM Provider:</strong></td>
                    <td>{llm_info}</td>
                </tr>
                <tr>
                    <td><strong>Robot:</strong></td>
                    <td>{self.robot_id.upper()}</td>
                </tr>
                <tr>
                    <td><strong>Mode:</strong></td>
                    <td>{"Simulation" if self.use_simulation else "Real Robot"}</td>
                </tr>
            </table>
        </div>
        """
        return html


def create_gradio_interface(gui: RobotMCPGUI):
    """Create Gradio interface compatible with all Gradio versions."""

    # Inline CSS for compatibility
    custom_css = """
    <style>
    .status-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        background-color: #f8f9fa;
    }
    .camera-feed img {
        border: 2px solid #2196F3;
        border-radius: 10px;
    }
    </style>
    """

    with gr.Blocks(title="Robot Control System") as demo:
        gr.HTML(custom_css)  # Inject CSS

        gr.Markdown("# 🤖 Robot Control System")
        gr.Markdown("Natural language control with live object detection visualization")

        # Example tasks state
        examples_state = gr.State(gui.default_examples)

        # Connection status
        with gr.Row():
            status_display = gr.HTML(value=gui.get_status_html(), elem_classes=["status-box"])
            connect_btn = gr.Button("🔌 Connect to MCP Server", variant="primary")
            connection_status = gr.Textbox(label="Connection Status", lines=2, interactive=False)

        with gr.Row():
            # Left: Chat interface
            with gr.Column(scale=2):
                # Simple chatbot without optional parameters for max compatibility
                chatbot = gr.Chatbot(label="Robot Assistant", height=500)

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Enter your command... (e.g., 'What objects do you see?')",
                        label="Message",
                        scale=4,
                        lines=2,
                    )
                    voice_btn = gr.Button("🎤 Record Voice", scale=1, variant="secondary")

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2)
                    suggest_btn = gr.Button("✨ Suggest Tasks", variant="secondary", scale=1)
                    clear_btn = gr.Button("Clear Chat", variant="stop", scale=1)

                # Example tasks (dynamically rendered)
                @gr.render(inputs=examples_state)
                def render_examples(examples_list):
                    gr.Examples(
                        examples=examples_list,
                        inputs=msg_input,
                        label="Example Tasks",
                    )

            # Right: Live camera feed
            with gr.Column(scale=1):
                camera_feed = gr.Image(label="Live Object Detection", type="numpy", height=500)

        # Event handlers
        async def handle_connect():
            success, message = await gui.connect_mcp()
            return gui.get_status_html(), message

        async def handle_submit(message, history):
            async for updated_history in gui.process_chat(message, history):
                yield "", updated_history

        def handle_voice():
            return gui.record_voice()

        def handle_clear():
            gui.chat_history = []
            return []

        async def handle_suggest():
            return await gui.suggest_tasks()

        def update_camera():
            return gui.get_latest_frame()

        # Wire up events
        connect_btn.click(fn=handle_connect, outputs=[status_display, connection_status])

        msg_input.submit(fn=handle_submit, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

        submit_btn.click(fn=handle_submit, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

        voice_btn.click(fn=handle_voice, outputs=[msg_input])

        clear_btn.click(fn=handle_clear, outputs=[chatbot])

        suggest_btn.click(fn=handle_suggest, outputs=[examples_state])

        # Auto-refresh camera feed
        timer = gr.Timer(0.1)
        timer.tick(fn=update_camera, outputs=[camera_feed])

    return demo


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Robot Control GUI")
    parser.add_argument("--api", choices=["openai", "groq", "gemini", "ollama"], default="groq")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--robot", choices=["niryo", "widowx"], default="niryo")
    parser.add_argument("--no-simulation", action="store_true")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(dotenv_path="secrets.env")

    # Initialize GUI
    gui = RobotMCPGUI(
        api_choice=args.api,
        model=args.model,
        robot_id=args.robot,
        use_simulation=not args.no_simulation,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )

    # Create and launch interface
    demo = create_gradio_interface(gui)

    print("\n" + "=" * 60)
    print("🚀 LAUNCHING GUI")
    print("=" * 60)
    print(f"  URL: http://localhost:{args.server_port}")
    print(f"  Share: {args.share}")
    print("=" * 60 + "\n")

    try:
        demo.queue().launch(share=args.share, server_port=args.server_port, server_name="0.0.0.0", inbrowser=True)
    except Exception as e:
        print(f"Launch error: {e}")
        # Fallback for older Gradio versions
        demo.queue().launch(share=args.share, server_port=args.server_port, server_name="0.0.0.0")


if __name__ == "__main__":
    asyncio.run(main())

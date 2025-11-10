"""
Robot Control GUI with MCP Integration

Gradio-based GUI for controlling a pick-and-place robot using MCP (Model Context Protocol).
The GUI includes a chatbot, live camera feed, voice input, and status indicators.
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import gradio as gr
import torch
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO: macht eigentlich keinen Sinn speech2text in robot_environment zu haben, sollte besser in diesem package sein
from robot_environment.speech2text import Speech2Text
from client.fastmcp_groq_client import RobotFastMCPClient
from redis_robot_comm import RedisImageStreamer


class RobotMCPGUI:
    """Main GUI class for robot control with MCP integration."""

    def __init__(
        self,
        groq_api_key: str,
        elevenlabs_api_key: str,
        model: str = "moonshotai/kimi-k2-instruct-0905",
        robot_id: str = "niryo",
        use_simulation: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the Robot MCP GUI.

        Args:
            groq_api_key: Groq API key for LLM
            elevenlabs_api_key: ElevenLabs API key for TTS
            model: Groq model to use
            robot_id: Robot type (niryo/widowx)
            use_simulation: Use simulation mode
            verbose: Enable verbose output
        """
        self.groq_api_key = groq_api_key
        # TODO: elevenlabs api key not used anymore, only in mcp server
        self.elevenlabs_api_key = elevenlabs_api_key
        self.model = model
        self.robot_id = robot_id
        self.use_simulation = use_simulation
        self.verbose = verbose

        # Status flags
        self.mcp_server_running = False
        self.mcp_client_connected = False
        self.server_process = None
        self.camera_thread = None

        # Initialize components
        # self.environment: Optional[Environment] = None
        self.mcp_client: Optional[RobotFastMCPClient] = None
        self.speech2text: Optional[Speech2Text] = None

        self._initialize_redis_streamer()

        # Executor for async tasks
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Message queue for chat updates
        self.message_queue = Queue()

        # Current camera frame
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Chat history
        self.chat_history = []

    def _initialize_redis_streamer(self):
        """Initialize Redis image streamer."""
        try:
            self._streamer = RedisImageStreamer(stream_name="robot_camera")
        except Exception as e:
            if self.verbose:
                print(f"Redis streamer initialization failed: {e}")
            self._streamer = None

    def initialize_environment(self):
        """Initialize the robot environment."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            # TODO: das darf man hier nicht machen, da environment im mcp server läuft
            # print("Initializing robot environment...")
            # self.environment = Environment(
            #     el_api_key=self.elevenlabs_api_key,
            #     use_simulation=self.use_simulation,
            #     robot_id=self.robot_id,
            #     verbose=self.verbose,
            #     start_camera_thread=False
            # )

            # Initialize speech recognition
            self.speech2text = Speech2Text(
                device=device, torch_dtype=torch_dtype, use_whisper_mic=True, verbose=self.verbose
            )

            # Start camera updates
            self._start_camera_updates()

            print("✓ Environment initialized")
            return True

        except Exception as e:
            print(f"✗ Failed to initialize environment: {e}")
            import traceback

            traceback.print_exc()
            return False

    # TODO: einen imagestreamer hier initialisieren, der sich immer das
    # aktuelle bild über redis holt. dort wird aktuell aber nur das original frame veröffentlicht und noch nicht
    # das annotated image
    def _start_camera_updates(self):
        """Start camera update thread."""

        def camera_loop():
            while True:
                try:
                    result = self._streamer.get_latest_image()
                    if not result:
                        if self.verbose:
                            print("No image available from Redis")

                    img, metadata = result

                    with self.frame_lock:
                        self.current_frame = img
                    time.sleep(0.1)  # 10 FPS
                except Exception as e:
                    print(f"Camera thread error: {e}")

        self.camera_thread = threading.Thread(target=camera_loop, daemon=True)
        self.camera_thread.start()
        print("✓ Camera thread started")

    async def start_mcp_server(self):
        """Start the FastMCP server."""
        try:
            print("Starting MCP server...")

            # Start server in separate process
            import subprocess

            server_path = Path(__file__).parent.parent / "server" / "fastmcp_robot_server.py"

            cmd = [
                sys.executable,
                str(server_path),
                "--robot",
                self.robot_id,
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]

            if not self.use_simulation:
                cmd.append("--no-simulation")

            if self.verbose:
                cmd.append("--verbose")

            self.server_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for server to start and check if it's actually running
            print("Waiting for server to initialize...")
            await asyncio.sleep(5)  # Erhöhe auf 5 Sekunden

            # Test if server is responding
            import httpx

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://127.0.0.1:8000/sse", timeout=5.0)
                    # Server antwortet
                    self.mcp_server_running = True
                    print("✓ MCP server started and responding")
                    self._add_system_message("🟢 MCP Server started")
                    return True
            except:
                # Check if process is still running
                if self.server_process.poll() is None:
                    self.mcp_server_running = True
                    print("✓ MCP server process running")
                    self._add_system_message("🟢 MCP Server started")
                    return True
                else:
                    print("✗ MCP server failed to start")
                    stdout, stderr = self.server_process.communicate()
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    return False

        except Exception as e:
            print(f"✗ Failed to start MCP server: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def connect_mcp_client(self):
        """Connect the MCP client."""
        try:
            print("Connecting MCP client...")

            self.mcp_client = RobotFastMCPClient(groq_api_key=self.groq_api_key, model=self.model)

            await self.mcp_client.connect()

            self.mcp_client_connected = True
            print("✓ MCP client connected")
            self._add_system_message("🟢 MCP Client connected")

            # Show available tools
            tools = [t.name for t in self.mcp_client.available_tools]
            self._add_system_message(f"Available tools: {', '.join(tools[:5])}...")

            return True

        except Exception as e:
            print(f"✗ Failed to connect MCP client: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def initialize_mcp(self):
        """Initialize both MCP server and client."""
        # Start server
        # TODO: aktuell muss ich server separat starten
        server_ok = True  # await self.start_mcp_server()
        if not server_ok:
            return False

        # Connect client
        client_ok = await self.connect_mcp_client()
        return client_ok

    def _add_system_message(self, message: str):
        """Add a system message to chat history."""
        self.chat_history.append({"role": "assistant", "content": f"🤖 **System:** {message}"})

    def _add_user_message(self, message: str):
        """Add a user message to chat history."""
        self.chat_history.append({"role": "user", "content": message})

    def _add_assistant_message(self, message: str):
        """Add an assistant message to chat history."""
        self.chat_history.append({"role": "assistant", "content": message})

    def _add_tool_call_message(self, tool_name: str, arguments: Dict[str, Any]):
        """Add a tool call notification to chat history."""
        args_str = ", ".join([f"{k}={v}" for k, v in arguments.items()])
        self.chat_history.append(
            {"role": "assistant", "content": f"🔧 **Calling tool:** `{tool_name}({args_str})`"}
        )

    async def process_user_input(self, user_input: str):
        """Process user input through MCP client."""
        if not user_input or not user_input.strip():
            yield self.chat_history
            return

        if not self.mcp_client_connected:
            self._add_assistant_message(
                "⚠️ MCP client not connected. Please wait for initialization."
            )
            yield self.chat_history
            return

        # Add user message
        self._add_user_message(user_input)
        yield self.chat_history

        # Add processing message
        self._add_assistant_message("🤔 Processing your request...")
        yield self.chat_history

        try:
            # Process through MCP client with tool call tracking
            async for _ in self._chat_with_tool_tracking(user_input):
                yield self.chat_history  # Yield after each update

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.chat_history[-1]["content"] = error_msg
            yield self.chat_history

    async def _chat_with_tool_tracking(self, user_message: str):
        """
        Process chat with tool call tracking.
        Modified version of mcp_client.chat() that shows tool calls.
        """
        # Add user message to client history
        self.mcp_client.conversation_history.append({"role": "user", "content": user_message})

        max_iterations = 4
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Prepare messages
            messages = [
                {"role": "system", "content": self.mcp_client.system_prompt}
            ] + self.mcp_client.conversation_history

            # Call Groq API
            response = self.mcp_client.groq_client.chat.completions.create(
                model=self.mcp_client.model,
                messages=messages,
                tools=self.mcp_client._convert_tools_to_groq_format(),
                tool_choice="auto",
                max_tokens=4096,
                temperature=0.7,
            )

            assistant_message = response.choices[0].message

            # Check for tool calls
            if assistant_message.tool_calls:
                # Add assistant's tool call request to history
                self.mcp_client.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                # Show tool calls in GUI
                for tc in assistant_message.tool_calls:
                    import json

                    try:
                        arguments = json.loads(tc.function.arguments)
                    except:
                        arguments = {}

                    self._add_tool_call_message(tc.function.name, arguments)
                    yield  # Update GUI

                # Process tool calls
                tool_results = await self.mcp_client.process_tool_calls(
                    assistant_message.tool_calls
                )

                # Add tool results to history
                self.mcp_client.conversation_history.extend(tool_results)

                continue
            else:
                # No more tool calls, update with final response
                final_response = assistant_message.content or "Task completed."

                self.mcp_client.conversation_history.append(
                    {"role": "assistant", "content": final_response}
                )

                # Update last message with final response
                self.chat_history[-1]["content"] = f"🤖 {final_response}"
                yield  # Final update
                return  # No value - just exit generator

        # Max iterations reached
        self.chat_history[-1]["content"] = "⚠️ Maximum iterations reached. Task may be incomplete."
        yield

    def record_voice_input(self) -> str:
        """Record voice input and transcribe."""
        try:
            if not self.speech2text:
                return "⚠️ Speech recognition not initialized"

            self._add_system_message("🎤 Recording... Please speak now")

            # Record and transcribe
            transcription = self.speech2text.record_and_transcribe()

            if transcription:
                self._add_system_message(f'🎤 Transcribed: "{transcription}"')
                return transcription
            else:
                self._add_system_message("🎤 No speech detected")
                return ""

        except Exception as e:
            error_msg = f"❌ Voice input error: {str(e)}"
            self._add_system_message(error_msg)
            return ""

    def get_current_frame(self):
        """Get the current camera frame."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame
            else:
                # Return placeholder
                import numpy as np

                return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_status_html(self):
        """Get HTML status display."""
        server_status = "🟢 Running" if self.mcp_server_running else "🔴 Stopped"
        client_status = "🟢 Connected" if self.mcp_client_connected else "🔴 Disconnected"

        html = f"""
        <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
            <h3>System Status</h3>
            <p><strong>MCP Server:</strong> {server_status}</p>
            <p><strong>MCP Client:</strong> {client_status}</p>
            <p><strong>Robot:</strong> {self.robot_id.upper()}</p>
            <p><strong>Mode:</strong> {"Simulation" if self.use_simulation else "Real Robot"}</p>
        </div>
        """
        return html

    def cleanup(self):
        """Cleanup resources."""
        print("\nCleaning up...")

        # Disconnect MCP client
        if self.mcp_client:
            try:
                asyncio.run(self.mcp_client.disconnect())
            except:
                pass

        # Stop MCP server
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        print("✓ Cleanup complete")


def create_gradio_interface(gui: RobotMCPGUI):
    """Create the Gradio interface."""

    # Example tasks
    example_tasks = [
        "What objects do you see?",
        "Pick up the pencil and place it at [0.2, 0.1]",
        "Move the red cube to the right of the blue square",
        "Sort all objects by size from smallest to largest",
        "Arrange objects in a triangle pattern",
        "Pick all pens or pencils and place them left next to the chocolate bar",
        "Swap positions of the two largest objects",
        "Place the smallest object in the center of the workspace",
    ]

    # Custom CSS
    css = """
    .status-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .camera-feed {
        border: 2px solid #2196F3;
        border-radius: 10px;
    }
    """

    with gr.Blocks(css=css, title="Robot Control System", analytics_enabled=False) as demo:
        # Timer für periodische Updates
        update_timer = gr.Timer(1)

        gr.Markdown("# 🤖 Robot Control System with MCP")
        gr.Markdown(
            "Natural language control for pick-and-place robots using Model Context Protocol"
        )

        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=2):
                status_display = gr.HTML(value=gui.get_status_html(), elem_classes=["status-box"])

                chatbot = gr.Chatbot(value=[], type="messages", height=500, label="Robot Assistant")

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Enter your command here... (e.g., 'Pick up the red cube')",
                        label="Task Input",
                        scale=4,
                        lines=1,
                    )
                    voice_btn = gr.Button("🎤 Voice", scale=1)

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", scale=1)

                task_examples = gr.Dropdown(
                    choices=example_tasks,
                    label="Example Tasks",
                    value=None,
                    # interactive=True
                )

            # Right column: Camera feed
            with gr.Column(scale=1):
                camera_display = gr.Image(
                    type="numpy", label="Live Camera View", height=500, elem_classes=["camera-feed"]
                )

        # Event handlers

        def user_submit(user_message):
            """Handle user message submission."""
            if not user_message.strip():
                return "", gui.chat_history
            return "", gui.chat_history

        def clear_chat():
            """Clear chat history."""
            gui.chat_history = []
            return None, []

        def select_example(example):
            """Handle example selection."""
            return example if example else ""

        def update_displays():
            """Update camera feed and status."""
            return gui.get_current_frame(), gui.get_status_html()

        def handle_voice_input_sync():
            """Handle voice input button click (synchronous wrapper)."""
            try:
                return gui.record_voice_input()
            except Exception as e:
                print(f"Voice input error: {e}")
                return ""

        # Wire up events
        msg_submit = msg.submit(
            fn=user_submit, inputs=[msg], outputs=[msg, chatbot], api_name=False
        )

        msg_submit.then(fn=gui.process_user_input, inputs=[msg], outputs=[chatbot], api_name=False)

        btn_click = submit_btn.click(
            fn=user_submit, inputs=[msg], outputs=[msg, chatbot], api_name=False
        )

        btn_click.then(fn=gui.process_user_input, inputs=[msg], outputs=[chatbot], api_name=False)

        clear_btn.click(fn=clear_chat, inputs=None, outputs=[msg, chatbot], api_name=False)

        task_examples.change(
            fn=select_example, inputs=[task_examples], outputs=[msg], api_name=False
        )

        voice_btn.click(fn=handle_voice_input_sync, inputs=None, outputs=[msg], api_name=False)

        # Timer für automatische Updates
        update_timer.tick(fn=update_displays, outputs=[camera_display, status_display])

    return demo


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Robot Control GUI with MCP")
    parser.add_argument("--robot", choices=["niryo", "widowx"], default="niryo", help="Robot type")
    parser.add_argument("--no-simulation", action="store_true", help="Use real robot")
    parser.add_argument(
        "--model", default="moonshotai/kimi-k2-instruct-0905", help="Groq model to use"
    )
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(dotenv_path="secrets.env")

    groq_api_key = os.getenv("GROQ_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")

    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found in environment")
        sys.exit(1)

    # Initialize GUI
    print("=" * 60)
    print("ROBOT CONTROL GUI INITIALIZATION")
    print("=" * 60)

    gui = RobotMCPGUI(
        groq_api_key=groq_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        model=args.model,
        robot_id=args.robot,
        use_simulation=not args.no_simulation,
        verbose=True,
    )

    # Initialize environment
    if not gui.initialize_environment():
        print("❌ Failed to initialize environment")
        sys.exit(1)

    # Initialize MCP
    if not await gui.initialize_mcp():
        print("❌ Failed to initialize MCP")
        sys.exit(1)

    # Create and launch Gradio interface
    demo = create_gradio_interface(gui)

    print("\n" + "=" * 60)
    print("🚀 LAUNCHING GUI")
    print("=" * 60)

    try:
        demo.queue().launch(share=args.share)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        gui.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

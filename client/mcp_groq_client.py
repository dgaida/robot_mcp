"""
MCP Client with Groq API for Robot Control

This client connects to the MCP robot server and uses Groq's LLM API
to process natural language commands and control the robot.

Install dependencies:
    pip install mcp groq

Usage:
    python mcp_groq_client.py

Set environment variable:
    export GROQ_API_KEY="your_groq_api_key"
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Client can use stdout since it doesn't communicate via stdio
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RobotMCPClient")


class RobotMCPClient:
    """MCP Client that uses Groq LLM for robot control."""

    def __init__(
        self,
        groq_api_key: str,
        model: str = "moonshotai/kimi-k2-instruct-0905",
        server_script_path: str = os.path.abspath(os.path.join("server", "mcp_robot_server.py")),
        robot_id: str = "niryo",
        use_simulation: bool = False,
    ):
        """
        Initialize the MCP client with Groq.

        Args:
            groq_api_key: Groq API key
            model: Groq model to use (default: llama-3.3-70b-versatile)
            server_script_path: Path to the MCP server script
            robot_id: Robot type ("niryo" or "widowx")
            use_simulation: Whether to use simulation mode
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        self.server_script_path = server_script_path
        self.robot_id = robot_id
        self.use_simulation = use_simulation

        self.session: Optional[ClientSession] = None
        self._context = None
        self.available_tools: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, str]] = []

        # System prompt for the LLM
        self.system_prompt = """You are a helpful robot control assistant. You have access to various tools to control a robotic arm and detect objects in its workspace.

Key capabilities:
- Pick and place objects using coordinates
- Detect and query objects in the workspace
- Move the robot to observation poses
- Get information about workspaces

When the user asks you to do something:
1. Use the available tools to accomplish the task
2. Always call get_detected_objects first to understand what's in the workspace
3. Use exact coordinates from detected objects for pick and place operations
4. Provide clear feedback about what you're doing
5. If something fails, explain what went wrong

Coordinate system:
- Coordinates are in meters [x, y]
- X-axis: forward/backward from robot base
- Y-axis: left/right from robot base

Location options for placement:
- "left next to" - places object to the left
- "right next to" - places object to the right
- "above" - places above
- "below" - places below
- "on top of" - stacks on top
- "close to" - near the coordinate

Always be precise and verify object positions before attempting to manipulate them."""

    async def connect(self):
        """Connect to the MCP server."""
        logger.info("Starting connection to MCP server...")

        self.server_script_path = os.path.abspath(os.path.join("server", "mcp_min_server.py"))

        server_params = StdioServerParameters(
            command="python",
            args=[
                self.server_script_path,
                self.robot_id,
                "true" if self.use_simulation else "false",
            ],
        )

        logger.debug(f"Server params: {server_params}")
        print("🤖 Connecting to MCP server...")
        print(f"   Robot: {self.robot_id}")
        print(f"   Simulation: {self.use_simulation}")
        print(f"   path2server: {self.server_script_path}")

        server_script = Path(__file__).parent / "server" / "mcp_robot_server.py"
        print(f"Server script: {server_script}")
        print(f"Exists: {server_script.exists()}")
        print(f"Exists: {Path(self.server_script_path).exists()}")
        print()

        try:
            # Use async with properly
            logger.debug("Creating stdio_client context...")
            self._stdio_context = stdio_client(server_params)

            logger.debug("Entering stdio_client context...")
            self.stdio, self.write = await self._stdio_context.__aenter__()

            print(f"   Read stream: {self.stdio}")
            print(f"   Write stream: {self.write}")
            logger.info("stdio_client connected")

            self.session = ClientSession(self.stdio, self.write)
            print(self.session)
            logger.info("ClientSession created")

            logger.debug("Initializing session with timeout...")
            await asyncio.wait_for(self.session.initialize(), timeout=30.0)
            logger.info("Session initialized")

            logger.debug("Requesting tool list with timeout...")
            tools_result = await asyncio.wait_for(self.session.list_tools(), timeout=10.0)
            logger.info(f"Received {len(tools_result.tools)} tools")

            self.available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools_result.tools
            ]

            logger.info(f"Tool names: {[t['function']['name'] for t in self.available_tools]}")

            logger.info(f"✓ Connected! Found {len(self.available_tools)} tools\n")

        except asyncio.TimeoutError as e:
            logger.error(f"Connection timed out - server not responding: {e}")
            print("✗ Connection timed out - check server logs")
            raise
        except Exception as e:
            logger.error(f"Connection failed: {e}", exc_info=True)
            print(f"✗ Connection failed: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            # ClientSession hat keine close() Methode
            # Die Session wird automatisch geschlossen wenn der Context Manager endet
            pass

        # Clean up the context manager
        if hasattr(self, "_stdio_context") and self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        print("\n✓ Disconnected from MCP server")

    def _convert_tools_to_groq_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to Groq's expected format."""
        return self.available_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Call a tool via MCP.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        print(f"🔧 Calling tool: {tool_name}")
        print(f"   Arguments: {json.dumps(arguments, indent=2)}")

        try:
            result = await self.session.call_tool(tool_name, arguments)

            # Extract text content from result
            if result.content:
                text_results = [item.text for item in result.content if hasattr(item, "text")]
                result_text = "\n".join(text_results)
                print(f"✓ Result: {result_text}\n")
                return result_text
            else:
                return "Tool executed successfully (no output)"

        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            print(f"✗ {error_msg}\n")
            return error_msg

    async def process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Process tool calls from the LLM response.

        Args:
            tool_calls: List of tool calls from LLM

        Returns:
            List of tool results for the next LLM call
        """
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            # Parse arguments
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # Call the tool via MCP
            result = await self.call_tool(tool_name, arguments)

            # Format result for Groq
            tool_results.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": result})

        return tool_results

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.

        Args:
            user_message: User's input message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Prepare messages for Groq
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

            # Call Groq API
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._convert_tools_to_groq_format(),
                    tool_choice="auto",
                    max_tokens=4096,
                    temperature=0.7,
                )

                assistant_message = response.choices[0].message

                # Check if the model wants to call tools
                if assistant_message.tool_calls:
                    # Add assistant's tool call request to history
                    self.conversation_history.append(
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

                    # Process tool calls
                    tool_results = await self.process_tool_calls(assistant_message.tool_calls)

                    # Add tool results to history
                    self.conversation_history.extend(tool_results)

                    # Continue loop to get final response
                    continue

                else:
                    # No more tool calls, return final response
                    final_response = assistant_message.content or "I completed the task."

                    # Add to history
                    self.conversation_history.append({"role": "assistant", "content": final_response})

                    return final_response

            except Exception as e:
                error_msg = f"Error calling Groq API: {str(e)}"
                print(f"✗ {error_msg}")
                return error_msg

        return "Maximum iterations reached. Task may be incomplete."

    def print_available_tools(self):
        """Print all available tools."""
        print("\n📋 Available Tools:")
        print("=" * 60)
        for tool in self.available_tools:
            func = tool["function"]
            print(f"\n🔧 {func['name']}")
            print(f"   {func['description'][:80]}...")
        print("=" * 60 + "\n")

    async def interactive_mode(self):
        """Run interactive chat mode."""
        print("\n" + "=" * 60)
        print("🤖 ROBOT CONTROL ASSISTANT")
        print("=" * 60)
        print("\nType your commands in natural language.")
        print("Examples:")
        print("  - 'What objects do you see?'")
        print("  - 'Pick up the pencil and place it at [0.2, 0.1]'")
        print("  - 'Move the red cube to the right of the blue square'")
        print("  - 'Show me the largest object'")
        print("\nType 'quit' or 'exit' to stop.")
        print("Type 'tools' to see available tools.")
        print("Type 'clear' to clear conversation history.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "tools":
                    self.print_available_tools()
                    continue

                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("✓ Conversation history cleared.\n")
                    continue

                print()  # Empty line for readability

                # Process the message
                response = await self.chat(user_input)

                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")

    async def run_command(self, command: str) -> str:
        """
        Run a single command and return the response.

        Args:
            command: Natural language command

        Returns:
            Assistant's response
        """
        response = await self.chat(command)
        return response


async def main():
    """Main entry point."""
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="MCP Client with Groq for Robot Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with real robot
  python mcp_groq_client.py --robot niryo

  # Interactive mode with simulation
  python mcp_groq_client.py --robot niryo --simulation

  # Single command
  python mcp_groq_client.py --command "What objects do you see?"

  # Different Groq model
  python mcp_groq_client.py --model llama-3.1-70b-versatile

Available Groq Models:
  - llama-3.3-70b-versatile (default, recommended)
  - llama-3.1-70b-versatile
  - llama-3.1-8b-instant (faster, less capable)
  - mixtral-8x7b-32768
        """,
    )

    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model to use (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--server",
        default="mcp_robot_server.py",
        help="Path to MCP server script (default: mcp_robot_server.py)",
    )
    parser.add_argument("--robot", choices=["niryo", "widowx"], default="niryo", help="Robot type (default: niryo)")
    parser.add_argument("--simulation", action="store_true", help="Use simulation mode")
    parser.add_argument("--command", help="Single command to execute (non-interactive mode)")

    args = parser.parse_args()

    load_dotenv(dotenv_path="secrets.env")

    # Get API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key required.")
        print("Set GROQ_API_KEY environment variable or use --api-key")
        print("\nGet your API key at: https://console.groq.com/keys")
        sys.exit(1)

    # Create client
    client = RobotMCPClient(
        groq_api_key=api_key,
        model=args.model,
        server_script_path=args.server,
        robot_id=args.robot,
        use_simulation=args.simulation,
    )

    try:
        # Connect to MCP server
        await client.connect()

        if args.command:
            # Single command mode
            print(f"You: {args.command}\n")
            response = await client.run_command(args.command)
            print(f"\n🤖 Assistant: {response}\n")
        else:
            # Interactive mode
            await client.interactive_mode()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

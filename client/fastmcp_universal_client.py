# fastmcp_universal_client.py
"""
Universal FastMCP Client with Multi-LLM Support

This client uses the integrated LLMClient to support multiple LLM providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- Groq (Llama, Mixtral, Kimi, Gemma)
- Google Gemini (Gemini 2.0/2.5)
- Ollama (Local models)
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from llm_client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RobotUniversalMCPClient")


class RobotUniversalMCPClient:
    """Universal MCP Client supporting multiple LLM providers.

    This client integrates with FastMCP server and uses LLMClient for
    flexible LLM provider selection (OpenAI, Groq, Gemini, Ollama).

    Attributes:
        llm_client: LLMClient instance for LLM interactions
        available_tools: List of tools exposed by MCP server
        conversation_history: Chat history for context
        client: FastMCP client instance
    """

    def __init__(
        self,
        api_choice: Literal["openai", "groq", "gemini", "ollama"] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize the Universal MCP Client.

        Args:
            api_choice: LLM provider ('openai', 'groq', 'gemini', 'ollama')
                       If None, auto-detected based on available API keys.
            model: Specific model name. If None, uses provider default.
            temperature: Sampling temperature (0.0-2.0). Default: 0.7.
            max_tokens: Maximum tokens to generate. Default: 4096.

        Examples:
            >>> # Auto-detect API based on available keys
            >>> client = RobotUniversalMCPClient()

            >>> # Explicit OpenAI selection
            >>> client = RobotUniversalMCPClient(
            ...     api_choice="openai",
            ...     model="gpt-4o"
            ... )

            >>> # Use Gemini
            >>> client = RobotUniversalMCPClient(
            ...     api_choice="gemini",
            ...     model="gemini-2.0-flash"
            ... )
        """
        # Initialize LLM client with multi-provider support
        self.llm_client = LLMClient(
            llm=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_choice=api_choice,
        )

        self.available_tools: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, str]] = []

        # System prompt for robot control
        self.system_prompt = """You are a helpful robot control assistant. You have access to various tools to control a robotic arm and detect objects in its workspace.

Robot Information:
1. The robot has a gripper that can pick objects up to 0.05 meters in size.
2. The robot has a gripper-mounted camera for workspace observation.
3. Detected objects include (x,y) world coordinates and size in meters.
4. **World Coordinate System**:
   - X-axis: vertical (increases bottom to top)
   - Y-axis: horizontal (increases right to left, origin at center)
   - Units: meters
5. **Workspace Boundaries**:
   - Upper left: (0.337, 0.087)
   - Lower right: (0.163, -0.087)
   - Negative y = right side, Positive y = left side

Key capabilities:
- Pick and place objects using coordinates
- Detect and query objects in workspace
- Move robot to observation poses
- Get workspace information

Guidelines:
1. If task is not in English, translate to English first
2. Use available tools to accomplish tasks
3. Always call get_detected_objects first before pick/place
4. Use exact coordinates from detected objects
5. Double-check object locations for similar objects
6. Match object names EXACTLY as returned by detection
7. Adhere to tool call format strictly
8. Provide clear feedback about actions
9. Explain failures when they occur
10. Always respond in English
11. Call move2observation_pose after completing tasks

Location options for placement:
- "left next to" - places left
- "right next to" - places right
- "above" - places above (farther in X)
- "below" - places below (closer in X)
- "on top of" - stacks on top
- "close to" - near coordinate

Always verify object positions before manipulation."""

        # Initialize FastMCP transport
        transport = SSETransport("http://127.0.0.1:8000/sse")
        self.client = Client(transport)

        logger.info(f"Initialized with {self.llm_client}")

    async def connect(self):
        """Connect to FastMCP server and discover available tools."""
        print("🤖 Connecting to FastMCP server...")
        await self.client.__aenter__()

        self.available_tools = await self.client.list_tools()
        print(f"Connected! Using {self.llm_client.api_choice.upper()} API")
        print(f"Model: {self.llm_client.llm}")
        print(f"Found {len(self.available_tools)} tools: {[t.name for t in self.available_tools]}")

    async def disconnect(self):
        """Disconnect from MCP server."""
        if hasattr(self, "client"):
            await self.client.__aexit__(None, None, None)
        print("✓ Disconnected from MCP server")

    def _convert_tools_to_function_format(self) -> List[Dict[str, Any]]:
        """Convert FastMCP tools to function calling format.

        Returns:
            List of tool definitions in OpenAI function calling format.
        """
        tools = []
        for tool in self.available_tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
            )
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool via MCP.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as dictionary

        Returns:
            Tool result as string
        """
        print(f"🔧 Calling tool: {tool_name}")
        print(f"   Arguments: {json.dumps(arguments, indent=2)}")

        try:
            result = await self.client.call_tool(tool_name, arguments)

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
        """Process tool calls from LLM response.

        Args:
            tool_calls: List of tool call objects from LLM

        Returns:
            List of tool results for next LLM call
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

            # Format result for LLM (OpenAI-compatible format)
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result,
                }
            )

        return tool_results

    async def chat(self, user_message: str) -> str:
        """Process a user message and return assistant's response.

        This method handles the complete interaction loop:
        1. Sends user message to LLM
        2. Processes any tool calls requested by LLM
        3. Returns final response to user

        Args:
            user_message: User's input message

        Returns:
            Assistant's final response

        Examples:
            >>> client = RobotUniversalMCPClient()
            >>> await client.connect()
            >>> response = await client.chat("What objects do you see?")
            >>> print(response)
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Prepare messages for LLM
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

            # Call LLM API (works with OpenAI, Groq, Gemini via compatibility)
            try:
                # For Ollama, we need to handle differently (no function calling)
                if self.llm_client.api_choice == "ollama":
                    # Ollama doesn't support function calling yet
                    # Fall back to text-based instruction following
                    response_text = self.llm_client.chat_completion(messages)

                    # Add to history and return
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    return response_text

                else:
                    # OpenAI, Groq, Gemini - all support function calling
                    # Build function calling request
                    tools_formatted = self._convert_tools_to_function_format()

                    # Create a temporary client for function calling
                    # (LLMClient doesn't expose this directly)
                    if self.llm_client.api_choice in ["openai", "groq", "gemini"]:
                        response = self.llm_client.client.chat.completions.create(
                            model=self.llm_client.llm,
                            messages=messages,
                            tools=tools_formatted,
                            tool_choice="auto",
                            max_tokens=self.llm_client.max_tokens,
                            temperature=self.llm_client.temperature,
                        )

                        assistant_message = response.choices[0].message

                        # Check if model wants to call tools
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

                            # Continue loop for final response
                            continue

                        else:
                            # No more tool calls, return final response
                            final_response = assistant_message.content or "I completed the task."

                            # Add to history
                            self.conversation_history.append({"role": "assistant", "content": final_response})

                            return final_response

            except Exception as e:
                error_msg = f"Error calling LLM API: {str(e)}"
                print(f"✗ {error_msg}")
                logger.error(error_msg, exc_info=True)
                return error_msg

        return "Maximum iterations reached. Task may be incomplete."

    def print_available_tools(self):
        """Print all available tools."""
        print("\n📋 Available Tools:")
        print("=" * 60)
        for tool in self.available_tools:
            print(f"\n🔧 {tool.name}")
            print(f"   {tool.description[:80]}...")
        print("=" * 60 + "\n")

    async def interactive_mode(self):
        """Run interactive chat mode."""
        print("\n" + "=" * 60)
        print("🤖 ROBOT CONTROL ASSISTANT (Universal LLM)")
        print("=" * 60)
        print(f"\nUsing: {self.llm_client.api_choice.upper()} - {self.llm_client.llm}")
        print("\nType your commands in natural language.")
        print("Examples:")
        print("  - 'What objects do you see?'")
        print("  - 'Pick up the pencil and place it at [0.2, 0.1]'")
        print("  - 'Move the red cube to the right of the blue square'")
        print("  - 'Show me the largest object'")
        print("\nType 'quit' or 'exit' to stop.")
        print("Type 'tools' to see available tools.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'switch' to switch LLM provider.")
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

                if user_input.lower() == "switch":
                    print("\n🔄 Current provider:", self.llm_client.api_choice.upper())
                    print("Available: openai, groq, gemini, ollama")
                    new_api = input("Switch to (or press Enter to cancel): ").strip().lower()

                    if new_api in ["openai", "groq", "gemini", "ollama"]:
                        try:
                            self.llm_client = LLMClient(
                                api_choice=new_api,
                                temperature=self.llm_client.temperature,
                                max_tokens=self.llm_client.max_tokens,
                            )
                            print(f"✓ Switched to {new_api.upper()} - {self.llm_client.llm}\n")
                        except Exception as e:
                            print(f"✗ Failed to switch: {e}\n")
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
                logger.error("Interactive mode error", exc_info=True)

    async def run_command(self, command: str) -> str:
        """Run a single command and return response.

        Args:
            command: Natural language command

        Returns:
            Assistant's response
        """
        response = await self.chat(command)
        return response


async def main():
    """Main entry point."""
    load_dotenv(dotenv_path="secrets.env")

    import argparse

    parser = argparse.ArgumentParser(
        description="Universal MCP Client with Multi-LLM Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect API (prefers OpenAI > Groq > Gemini > Ollama)
  python client/fastmcp_universal_client.py

  # Explicit OpenAI
  python client/fastmcp_universal_client.py --api openai --model gpt-4o

  # Use Groq
  python client/fastmcp_universal_client.py --api groq

  # Use Gemini
  python client/fastmcp_universal_client.py --api gemini --model gemini-2.0-flash

  # Use local Ollama
  python client/fastmcp_universal_client.py --api ollama --model llama3.2:1b

  # Single command mode
  python client/fastmcp_universal_client.py --command "What objects do you see?"

Supported Providers:
  OpenAI:  gpt-4o, gpt-4o-mini, gpt-3.5-turbo
  Groq:    moonshotai/kimi-k2-instruct-0905, llama-3.3-70b-versatile
  Gemini:  gemini-2.0-flash, gemini-2.5-pro
  Ollama:  llama3.2:1b, mistral, codellama
        """,
    )

    parser.add_argument(
        "--api",
        choices=["openai", "groq", "gemini", "ollama"],
        help="LLM provider (auto-detected if not specified)",
    )
    parser.add_argument("--model", help="Specific model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens (default: 4096)")
    parser.add_argument("--command", help="Single command to execute (non-interactive)")

    args = parser.parse_args()

    # Create client
    client = RobotUniversalMCPClient(
        api_choice=args.api,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
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

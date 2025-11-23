# fastmcp_universal_client.py
"""
Universal FastMCP Client with Multi-LLM Support and Chain-of-Thought Prompting

This client uses the integrated LLMClient to support multiple LLM providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- Groq (Llama, Mixtral, Kimi, Gemma)
- Google Gemini (Gemini 2.0/2.5)
- Ollama (Local models)

Features chain-of-thought prompting where the LLM explicitly states:
1. Understanding of the task
2. Planning steps before execution
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from llm_client import LLMClient

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Create log directory if it doesn't exist
os.makedirs("log", exist_ok=True)

# Unique log file for this client session
log_filename = os.path.join("log", f'mcp_client_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging - file only (no console to keep UI clean)
# Use UTF-8 encoding to handle emojis on Windows
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
    force=True,
)

logger = logging.getLogger("RobotUniversalMCPClient")

# Also enable logging for LLM client
logging.getLogger("llm_client").setLevel(logging.INFO)

logger.info("=" * 80)
logger.info(f"Universal MCP Client starting - Log file: {log_filename}")
logger.info("=" * 80)


class RobotUniversalMCPClient:
    """Universal MCP Client supporting multiple LLM providers with Chain-of-Thought.

    This client integrates with FastMCP server and uses LLMClient for
    flexible LLM provider selection (OpenAI, Groq, Gemini, Ollama).

    Features chain-of-thought prompting for better transparency and reasoning.

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
        logger.info("=" * 60)
        logger.info("CLIENT INITIALIZATION")
        logger.info(f"  API Choice: {api_choice or 'auto-detect'}")
        logger.info(f"  Model: {model or 'default'}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Max Tokens: {max_tokens}")
        logger.info("=" * 60)

        # Initialize LLM client with multi-provider support
        self.llm_client = LLMClient(
            llm=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_choice=api_choice,
        )

        logger.info(f"Initialized with {self.llm_client}")
        logger.info(f"  Provider: {self.llm_client.api_choice.upper()}")
        logger.info(f"  Model: {self.llm_client.llm}")

        self.available_tools: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, str]] = []

        # Enhanced system prompt with chain-of-thought instructions
        self.system_prompt = """You are a helpful robot control assistant with explicit reasoning capabilities. You have access to various tools to control a robotic arm and detect objects in its workspace.

**CRITICAL: Chain-of-Thought Reasoning Protocol**

When the user gives you a task, you MUST follow this two-phase approach:

**PHASE 1: PLANNING (Required before any tool calls)**
Before calling ANY tools, you must explicitly state:
1. **Task Understanding**: Restate the user's goal in your own words
2. **Analysis**: Break down what information you need and what actions are required
3. **Execution Plan**: List the specific tools you will call and in what order

Format your planning response like this:
"🎯 Task Understanding: [restate goal]
📋 Analysis: [what's needed]
🔧 Execution Plan:
   Step 1: [tool_name] - [why]
   Step 2: [tool_name] - [why]
   ..."

**PHASE 2: EXECUTION**
After stating your plan, I will prompt you to proceed. Then you MUST use the function calling API (NOT text-based tool tags).

**CRITICAL SPATIAL RULES:**
1. **NO OVERLAPPING OBJECTS**: You CANNOT place an object where another object currently exists!
2. **SWAP PROCEDURE**: When swapping two objects A and B:
   - Step 1: Find a FREE temporary location (use get_largest_free_space_with_center)
   - Step 2: Move A to temporary location
   - Step 3: Move B to A's original location (now empty)
   - Step 4: Move A from temporary to B's original location (now empty)
3. **ALWAYS CHECK**: Before placing, verify the target location is free of other objects
4. **FREE SPACE**: Use get_largest_free_space_with_center or check get_detected_objects to find empty areas

Example CORRECT swap plan:
"To swap pen and cube:
1. get_detected_objects - find current positions
2. get_largest_free_space_with_center - find temporary spot
3. pick_place_object - move pen to temporary spot
4. pick_place_object - move cube to pen's original spot (now empty)
5. pick_place_object - move pen from temporary to cube's original spot (now empty)"

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
1. ALWAYS start with PHASE 1 (Planning) before calling tools
2. If task is not in English, translate to English first
3. Always call get_detected_objects first before pick/place
4. Use exact coordinates from detected objects
5. Double-check object locations for similar objects
6. Match object names EXACTLY as returned by detection
7. Use the FUNCTION CALLING API - never use XML tags like <use_tool>
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

**REMINDER**: When swapping or moving objects, ALWAYS use a temporary free location first. Never try to place object A directly where object B is currently located!

Always verify object positions before manipulation."""

        # Initialize FastMCP transport
        transport = SSETransport("http://127.0.0.1:8000/sse")
        self.client = Client(transport)

        logger.info("Client initialization complete")

    async def connect(self):
        """Connect to FastMCP server and discover available tools."""
        logger.info("=" * 60)
        logger.info("CONNECTING TO SERVER")
        logger.info("  Server URL: http://127.0.0.1:8000/sse")
        logger.info("=" * 60)

        print("🤖 Connecting to FastMCP server...")
        await self.client.__aenter__()

        self.available_tools = await self.client.list_tools()

        tool_names = [t.name for t in self.available_tools]
        logger.info("Connected successfully")
        logger.info(f"  Provider: {self.llm_client.api_choice.upper()}")
        logger.info(f"  Model: {self.llm_client.llm}")
        logger.info(f"  Available tools ({len(tool_names)}): {', '.join(tool_names)}")

        print(f"Connected! Using {self.llm_client.api_choice.upper()} API")
        print(f"Model: {self.llm_client.llm}")
        print(f"Found {len(self.available_tools)} tools: {tool_names}")

    async def disconnect(self):
        """Disconnect from MCP server."""
        logger.info("=" * 60)
        logger.info("DISCONNECTING FROM SERVER")
        logger.info("=" * 60)

        if hasattr(self, "client"):
            await self.client.__aexit__(None, None, None)

        logger.info("Disconnected successfully")
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
        logger.info("-" * 60)
        logger.info(f"TOOL CALL: {tool_name}")
        logger.info(f"  Arguments: {json.dumps(arguments, indent=2)}")

        try:
            result = await self.client.call_tool(tool_name, arguments)

            # Extract text content from result
            if result.content:
                text_results = [item.text for item in result.content if hasattr(item, "text")]
                result_text = "\n".join(text_results)

                logger.info(f"  Result: {result_text}")
                logger.info("  Status: SUCCESS")
                logger.info("-" * 60)

                print(f"✓ Result: {result_text}\n")
                return result_text
            else:
                logger.info("  Result: Tool executed successfully (no output)")
                logger.info("  Status: SUCCESS")
                logger.info("-" * 60)
                return "Tool executed successfully (no output)"

        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            logger.error(f"  Error: {str(e)}", exc_info=True)
            logger.info("  Status: FAILED")
            logger.info("-" * 60)

            print(f"✗ {error_msg}\n")
            return error_msg

    async def process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Process tool calls from LLM response.

        Args:
            tool_calls: List of tool call objects from LLM

        Returns:
            List of tool results for next LLM call
        """
        logger.info(f"Processing {len(tool_calls)} tool call(s)")

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

    def _extract_planning_phase(self, content: str) -> tuple[bool, str]:
        """Extract and identify if response contains planning phase.

        Args:
            content: Assistant's response content

        Returns:
            Tuple of (is_planning_phase, planning_text)
        """
        # Check for planning indicators
        planning_indicators = ["🎯", "📋", "🔧", "Task Understanding:", "Analysis:", "Execution Plan:"]

        has_planning = any(indicator in content for indicator in planning_indicators)

        return has_planning, content

    async def chat(self, user_message: str) -> str:
        """Process a user message and return assistant's response.

        This method handles the complete interaction loop with chain-of-thought:
        1. ENFORCES planning phase: LLM must explain reasoning first
        2. Then allows tool calls after planning is complete
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
        logger.info("=" * 80)
        logger.info("NEW CHAT MESSAGE")
        logger.info(f"  User: {user_message}")
        logger.info("=" * 80)

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        max_iterations = 15  # Increased to accommodate planning phase
        iteration = 0
        planning_phase_complete = False

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"--- Iteration {iteration}/{max_iterations} ---")

            # Prepare messages for LLM
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

            # Call LLM API
            try:
                # For Ollama, handle differently (no function calling)
                if self.llm_client.api_choice == "ollama":
                    logger.info("Using Ollama (text-based mode)")

                    response_text = self.llm_client.chat_completion(messages)

                    # Check if this is planning phase
                    is_planning, planning_text = self._extract_planning_phase(response_text)

                    if is_planning and not planning_phase_complete:
                        logger.info("=" * 80)
                        logger.info("CHAIN-OF-THOUGHT: PLANNING PHASE")
                        logger.info(planning_text)
                        logger.info("=" * 80)

                        print("\n" + "=" * 70)
                        print("💭 CHAIN-OF-THOUGHT REASONING")
                        print("=" * 70)
                        print(planning_text)
                        print("=" * 70 + "\n")

                        planning_phase_complete = True

                    # Add to history and return
                    self.conversation_history.append({"role": "assistant", "content": response_text})

                    logger.info(f"Assistant response: {response_text}")
                    logger.info("=" * 80)

                    return response_text

                else:
                    # OpenAI, Groq, Gemini - all support function calling
                    logger.info(f"Using {self.llm_client.api_choice.upper()} with function calling")

                    tools_formatted = self._convert_tools_to_function_format()

                    # PHASE 1: FORCE PLANNING - Don't provide tools on first call
                    if not planning_phase_complete:
                        logger.info("PHASE 1: Requesting planning (tools disabled)")

                        # Call without tools to force text response (planning)
                        response = self.llm_client.client.chat.completions.create(
                            model=self.llm_client.llm,
                            messages=messages,
                            max_tokens=self.llm_client.max_tokens,
                            temperature=self.llm_client.temperature,
                        )

                        assistant_message = response.choices[0].message

                        if assistant_message.content:
                            planning_text = assistant_message.content

                            # Log and display the planning
                            logger.info("=" * 80)
                            logger.info("CHAIN-OF-THOUGHT: PLANNING PHASE")
                            logger.info(planning_text)
                            logger.info("=" * 80)

                            print("\n" + "=" * 70)
                            print("💭 CHAIN-OF-THOUGHT REASONING")
                            print("=" * 70)
                            print(planning_text)
                            print("=" * 70 + "\n")

                            # Add planning to history
                            self.conversation_history.append({"role": "assistant", "content": planning_text})

                            # Mark planning as complete
                            planning_phase_complete = True

                            # Add instruction to proceed with execution using function calling API
                            self.conversation_history.append(
                                {
                                    "role": "user",
                                    "content": "Good! Now execute your plan step by step. IMPORTANT: Use the function calling API to call tools (you will see the 'tools' parameter). Do NOT use XML tags or text-based tool syntax. Call one tool at a time and wait for results.",
                                }
                            )

                            logger.info("Planning phase complete. Proceeding to execution with function calling...")

                            # Continue to next iteration for tool execution
                            continue

                    # PHASE 2: EXECUTION - Now allow tool calls
                    logger.info("PHASE 2: Execution (tools enabled)")

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
                        logger.info(f"LLM requested {len(assistant_message.tool_calls)} tool call(s)")

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

                        logger.info(f"Final assistant response: {final_response}")
                        logger.info("=" * 80)

                        # Add to history
                        self.conversation_history.append({"role": "assistant", "content": final_response})

                        return final_response

            except Exception as e:
                error_msg = f"Error calling LLM API: {str(e)}"
                logger.error(error_msg, exc_info=True)
                logger.info("=" * 80)

                print(f"✗ {error_msg}")
                return error_msg

        logger.warning("Maximum iterations reached")
        logger.info("=" * 80)
        return "Maximum iterations reached. Task may be incomplete."

    def print_available_tools(self):
        """Print all available tools."""
        logger.info("Listing available tools")

        print("\n📋 Available Tools:")
        print("=" * 60)
        for tool in self.available_tools:
            print(f"\n🔧 {tool.name}")
            print(f"   {tool.description[:80]}...")
        print("=" * 60 + "\n")

    async def interactive_mode(self):
        """Run interactive chat mode."""
        logger.info("=" * 60)
        logger.info("INTERACTIVE MODE STARTED")
        logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("🤖 ROBOT CONTROL ASSISTANT (Universal LLM + CoT)")
        print("=" * 60)
        print(f"\nUsing: {self.llm_client.api_choice.upper()} - {self.llm_client.llm}")
        print(f"Log file: {log_filename}")
        print("\n✨ Chain-of-Thought Enabled ✨")
        print("The assistant will explain its reasoning before acting.")
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

                logger.info(f"User input: {user_input}")

                if user_input.lower() in ["quit", "exit", "q"]:
                    logger.info("User requested exit")
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "tools":
                    self.print_available_tools()
                    continue

                if user_input.lower() == "clear":
                    self.conversation_history = []
                    logger.info("Conversation history cleared")
                    print("✓ Conversation history cleared.\n")
                    continue

                if user_input.lower() == "switch":
                    logger.info("User requested provider switch")

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
                            logger.info(f"Switched to {new_api.upper()} - {self.llm_client.llm}")
                            print(f"✓ Switched to {new_api.upper()} - {self.llm_client.llm}\n")
                        except Exception as e:
                            logger.error(f"Failed to switch provider: {e}", exc_info=True)
                            print(f"✗ Failed to switch: {e}\n")
                    continue

                print()  # Empty line for readability

                # Process the message (will show chain-of-thought reasoning)
                response = await self.chat(user_input)

                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                logger.info("Interrupted by user (Ctrl+C)")
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}", exc_info=True)
                print(f"\n✗ Error: {e}\n")

        logger.info("=" * 60)
        logger.info("INTERACTIVE MODE ENDED")
        logger.info("=" * 60)

    async def run_command(self, command: str) -> str:
        """Run a single command and return response.

        Args:
            command: Natural language command

        Returns:
            Assistant's response
        """
        logger.info("=" * 60)
        logger.info("SINGLE COMMAND MODE")
        logger.info(f"  Command: {command}")
        logger.info("=" * 60)

        response = await self.chat(command)

        logger.info(f"Command completed. Response: {response}")
        logger.info("=" * 60)

        return response


async def main():
    """Main entry point."""
    load_dotenv(dotenv_path="secrets.env")

    import argparse

    parser = argparse.ArgumentParser(
        description="Universal MCP Client with Multi-LLM Support and Chain-of-Thought",
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

    logger.info("=" * 80)
    logger.info("MAIN ENTRY POINT")
    logger.info(f"  Arguments: {vars(args)}")
    logger.info("=" * 80)

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
            logger.info("Running in single command mode")
            print(f"You: {args.command}\n")
            response = await client.run_command(args.command)
            print(f"\n🤖 Assistant: {response}\n")
        else:
            # Interactive mode
            logger.info("Running in interactive mode")
            await client.interactive_mode()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.disconnect()
        logger.info("=" * 80)
        logger.info(f"CLIENT SESSION ENDED - Log saved to: {log_filename}")
        logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

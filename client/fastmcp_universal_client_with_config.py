# client/fastmcp_universal_client_with_config.py
"""
Universal FastMCP Client with Configuration Management and Redis Integration

This client uses ConfigManager for centralized settings and integrates with Redis
for video text overlays. Supports multiple LLM providers (OpenAI, Groq, Gemini, Ollama).

Features:
- Chain-of-thought prompting with explicit reasoning
- Multi-LLM support with auto-detection
- Configuration management
- Redis integration for video overlays
- Comprehensive logging
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from llm_client import LLMClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager, load_config

# Try importing Redis text overlay manager
HAS_TEXT_OVERLAY = False
try:
    from redis_robot_comm import RedisTextOverlayManager

    HAS_TEXT_OVERLAY = True
except ImportError:
    print("⚠️ redis_robot_comm not available - video text overlays disabled")


# ============================================================================
# LOGGING SETUP WITH CONFIG
# ============================================================================


def setup_logging(config: ConfigManager) -> logging.Logger:
    """Setup logging based on configuration."""
    log_dir = Path(config.server.log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mcp_client_{timestamp}.log"

    logging.basicConfig(
        level=config.server.log_level,
        format=config.logging.format,
        datefmt=config.logging.date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    # Set module-specific levels
    for module_name, level in config.logging.levels.items():
        logging.getLogger(module_name).setLevel(level)

    logger = logging.getLogger("RobotUniversalMCPClient")

    logger.info("=" * 80)
    logger.info(f"Universal MCP Client starting - Log file: {log_file}")
    logger.info("=" * 80)

    return logger


# ============================================================================
# CLIENT CLASS WITH CONFIG AND REDIS INTEGRATION
# ============================================================================


class RobotUniversalMCPClient:
    """
    Universal MCP Client with configuration management and Redis integration.

    This client integrates with FastMCP server and uses LLMClient for
    flexible LLM provider selection (OpenAI, Groq, Gemini, Ollama).

    Features:
    - Chain-of-thought prompting for transparency
    - Configuration-based settings
    - Redis integration for video text overlays
    - Comprehensive logging

    Attributes:
        config: ConfigManager instance
        llm_client: LLMClient instance for LLM interactions
        available_tools: List of tools exposed by MCP server
        conversation_history: Chat history for context
        client: FastMCP client instance
        text_overlay_manager: Redis text overlay manager (optional)
        logger: Logger instance
    """

    def __init__(
        self,
        config: ConfigManager,
        api_choice: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize client with configuration.

        Args:
            config: ConfigManager instance
            api_choice: Override default LLM provider
            model: Override default model

        Examples:
            >>> config = load_config()
            >>> client = RobotUniversalMCPClient(config)
            >>> await client.connect()
            >>> response = await client.chat("What objects do you see?")
        """
        self.config = config
        self.logger = logging.getLogger("RobotUniversalMCPClient")

        self.logger.info("=" * 60)
        self.logger.info("CLIENT INITIALIZATION")
        self.logger.info(f"  Config Provider: {config.llm.default_provider}")
        self.logger.info(f"  Override API: {api_choice or 'None'}")
        self.logger.info(f"  Override Model: {model or 'None'}")
        self.logger.info("=" * 60)

        # Determine LLM provider
        if api_choice is None:
            if config.llm.default_provider == "auto":
                api_choice = None  # Let LLMClient auto-detect
            else:
                api_choice = config.llm.default_provider

        # Determine model
        if model is None and api_choice:
            # Use configured default model for this provider
            if api_choice in config.llm.providers:
                model = config.llm.providers[api_choice].default_model

        # Initialize LLM client with config values
        self.llm_client = LLMClient(
            llm=model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_choice=api_choice,
        )

        self.logger.info(f"Initialized with {self.llm_client}")
        self.logger.info(f"  Provider: {self.llm_client.api_choice.upper()}")
        self.logger.info(f"  Model: {self.llm_client.llm}")

        self.available_tools: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize Redis text overlay manager
        self.text_overlay_manager = None
        if HAS_TEXT_OVERLAY:
            try:
                self.text_overlay_manager = RedisTextOverlayManager(host=config.redis.host, port=config.redis.port)
                self.logger.info("Redis text overlay manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize text overlay manager: {e}")

        # Build system prompt with workspace info from config
        self.system_prompt = self._build_system_prompt()

        # Initialize FastMCP transport with configured server
        server_url = f"http://{config.server.host}:{config.server.port}/sse"
        transport = SSETransport(server_url)
        self.client = Client(transport)

        self.logger.info(f"FastMCP client configured for: {server_url}")
        self.logger.info("Client initialization complete")

    def _build_system_prompt(self) -> str:
        """Build system prompt with configuration values."""
        # Get workspace bounds for current robot
        robot_type = self.config.robot.type
        workspace = self.config.robot.workspace.get(robot_type)

        if workspace:
            bounds_info = f"""
**Workspace Boundaries ({workspace.id})**:
   - Upper left: ({workspace.bounds.x_max:.3f}, {workspace.bounds.y_max:.3f})
   - Lower right: ({workspace.bounds.x_min:.3f}, {workspace.bounds.y_min:.3f})
   - Center: ({workspace.center[0]:.3f}, {workspace.center[1]:.3f})
   - Negative y = right side, Positive y = left side
"""
        else:
            bounds_info = ""

        # Get detection labels
        labels_str = ", ".join(self.config.detection.default_labels[:10])
        if len(self.config.detection.default_labels) > 10:
            labels_str += ", ..."

        # Build prompt with chain-of-thought settings
        cot_instructions = ""
        if self.config.llm.enable_cot:
            cot_instructions = """
**CRITICAL: Chain-of-Thought Reasoning Protocol**

When the user gives you a task, you MUST follow this two-phase approach:

**PHASE 1: PLANNING (Required before any tool calls)**
Before calling ANY tools, you must explicitly state:
1. **Task Understanding**: Restate the user's goal in your own words
2. **Analysis**: Break down what information you need and what actions are required
3. **Execution Plan**: List the specific tools you will call and in what order
   - IMPORTANT: Your plan MUST include move2observation_pose as the FINAL step

Format your planning response like this:
"🎯 Task Understanding: [restate goal]
📋 Analysis: [what's needed]
🔧 Execution Plan:
   Step 1: [tool_name] - [why]
   Step 2: [tool_name] - [why]
   ...
   Step N: move2observation_pose - return to observation position (REQUIRED)"

**PHASE 2: EXECUTION**
After stating your plan, proceed with tool calls using the function calling API.
"""

        prompt = f"""You are a helpful robot control assistant with explicit reasoning capabilities. You have access to various tools to control a robotic arm and detect objects in its workspace.

{cot_instructions}

**CRITICAL SPATIAL RULES:**
1. **NO OVERLAPPING OBJECTS**: You CANNOT place an object where another object currently exists!
2. **SWAP PROCEDURE**: When swapping two objects A and B:
   - Step 1: Find a FREE temporary location (use get_largest_free_space_with_center)
   - Step 2: Move A to temporary location
   - Step 3: Move B to A's original location (now empty)
   - Step 4: Move A from temporary to B's original location (now empty)
3. **ALWAYS CHECK**: Before placing, verify the target location is free of other objects
4. **FREE SPACE**: Use get_largest_free_space_with_center to find empty areas
Example CORRECT swap plan:
"To swap pen and cube:
1. get_detected_objects - find current positions
2. get_largest_free_space_with_center - find temporary spot
3. pick_place_object - move pen to temporary spot
4. pick_place_object - move cube to pen's original spot (now empty)
5. pick_place_object - move pen from temporary to cube's original spot (now empty)
6. move2observation_pose - return to observation position (REQUIRED)"

Robot Information:
1. The robot has a gripper that can pick objects up to 0.05 meters in size.
2. The robot has a gripper-mounted camera for workspace observation.
3. Detected objects include (x,y) world coordinates and size in meters.
4. **World Coordinate System**:
   - X-axis: vertical (increases bottom to top)
   - Y-axis: horizontal (increases right to left, origin at center)
   - Units: meters
5. {bounds_info}

**Detectable Objects**: {labels_str}

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
11. **MANDATORY FINAL STEP**: After completing ALL manipulation tasks (pick, place, push), you MUST call move2observation_pose with the workspace_id to return the robot to its home observation position. This is NOT optional - every task sequence must end with this call. Your task is not complete until the robot is back at observation pose.

Location options for placement:
- "left next to" - places left
- "right next to" - places right
- "above" - places above (farther in X)
- "below" - places below (closer in X)
- "on top of" - stacks on top
- "close to" - near coordinate

**REMINDER**:
- When swapping or moving objects, ALWAYS use a temporary free location first. Never try to place object A directly where object B is currently located!
- Every task MUST end with move2observation_pose(workspace_id) - no exceptions. Include this in your plan and execute it as your final action.

Always verify object positions before manipulation."""

        return prompt

    async def connect(self):
        """Connect to FastMCP server and discover available tools."""
        self.logger.info("=" * 60)
        self.logger.info("CONNECTING TO SERVER")
        self.logger.info(f"  Server: {self.config.server.host}:{self.config.server.port}")
        self.logger.info("=" * 60)

        print(f"🤖 Connecting to FastMCP server at {self.config.server.host}:{self.config.server.port}...")
        await self.client.__aenter__()

        self.available_tools = await self.client.list_tools()

        tool_names = [t.name for t in self.available_tools]
        self.logger.info("Connected successfully")
        self.logger.info(f"  Provider: {self.llm_client.api_choice.upper()}")
        self.logger.info(f"  Model: {self.llm_client.llm}")
        self.logger.info(f"  Available tools ({len(tool_names)}): {', '.join(tool_names)}")

        print(f"✓ Connected! Using {self.llm_client.api_choice.upper()} API")
        print(f"  Model: {self.llm_client.llm}")
        print(f"  Found {len(self.available_tools)} tools: {tool_names}")

    async def disconnect(self):
        """Disconnect from MCP server."""
        self.logger.info("=" * 60)
        self.logger.info("DISCONNECTING FROM SERVER")
        self.logger.info("=" * 60)

        if hasattr(self, "client"):
            await self.client.__aexit__(None, None, None)

        self.logger.info("Disconnected successfully")
        print("✓ Disconnected from MCP server")

    def _convert_tools_to_function_format(self) -> List[Dict[str, Any]]:
        """
        Convert FastMCP tools to function calling format.

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
        """
        Call a tool via MCP.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as dictionary

        Returns:
            Tool result as string
        """
        self.logger.info("-" * 60)
        self.logger.info(f"TOOL CALL: {tool_name}")
        self.logger.info(f"  Arguments: {json.dumps(arguments, indent=2)}")

        try:
            result = await self.client.call_tool(tool_name, arguments)

            # Extract text content from result
            if result.content:
                text_results = [item.text for item in result.content if hasattr(item, "text")]
                result_text = "\n".join(text_results)

                self.logger.info(f"  Result: {result_text}")
                self.logger.info("  Status: SUCCESS")
                self.logger.info("-" * 60)

                print(f"✓ Result: {result_text}\n")
                return result_text
            else:
                self.logger.info("  Result: Tool executed successfully (no output)")
                self.logger.info("  Status: SUCCESS")
                self.logger.info("-" * 60)
                return "Tool executed successfully (no output)"

        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            self.logger.error(f"  Error: {str(e)}", exc_info=True)
            self.logger.info("  Status: FAILED")
            self.logger.info("-" * 60)

            print(f"✗ {error_msg}\n")
            return error_msg

    async def process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Process tool calls from LLM response.

        Args:
            tool_calls: List of tool call objects from LLM

        Returns:
            List of tool results for next LLM call
        """
        self.logger.info(f"Processing {len(tool_calls)} tool call(s)")

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
        """
        Extract and identify if response contains planning phase.

        Args:
            content: Assistant's response content

        Returns:
            Tuple of (is_planning_phase, planning_text)
        """
        # Check for planning indicators
        planning_indicators = ["🎯", "📋", "🔧", "Task Understanding:", "Analysis:", "Execution Plan:"]

        has_planning = any(indicator in content for indicator in planning_indicators)

        return has_planning, content

    async def _publish_user_task_to_redis(self, user_message: str):
        """
        Publish user task to Redis for video overlay.

        Args:
            user_message: The user's command/task
        """
        if self.text_overlay_manager:
            try:
                self.text_overlay_manager.publish_user_task(
                    task=user_message, metadata={"timestamp": datetime.now().isoformat(), "client": "universal_client"}
                )
                self.logger.info(f"Published user task to Redis: {user_message}")
            except Exception as e:
                self.logger.warning(f"Failed to publish user task to Redis: {e}")

    async def _call_set_user_task_tool(self, user_message: str):
        """
        Call the set_user_task MCP tool.

        Args:
            user_message: The user's command/task
        """
        try:
            # Check if set_user_task tool is available
            tool_names = [t.name for t in self.available_tools]

            if "set_user_task" in tool_names:
                await self.call_tool("set_user_task", {"task": user_message})
                self.logger.info(f"Called set_user_task tool: {user_message}")
            else:
                # Fallback to direct Redis publish
                self.logger.info("set_user_task tool not available, using direct Redis publish")
                await self._publish_user_task_to_redis(user_message)

        except Exception as e:
            self.logger.warning(f"Failed to call set_user_task tool: {e}")
            # Fallback to direct Redis publish
            await self._publish_user_task_to_redis(user_message)

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return assistant's response.

        This method handles the complete interaction loop with chain-of-thought:
        1. Publishes user task to Redis for video overlay
        2. ENFORCES planning phase: LLM must explain reasoning first
        3. Then allows tool calls after planning is complete
        4. Returns final response to user

        Args:
            user_message: User's input message

        Returns:
            Assistant's final response

        Examples:
            >>> client = RobotUniversalMCPClient(config)
            >>> await client.connect()
            >>> response = await client.chat("What objects do you see?")
            >>> print(response)
        """
        self.logger.info("=" * 80)
        self.logger.info("NEW CHAT MESSAGE")
        self.logger.info(f"  User: {user_message}")
        self.logger.info("=" * 80)

        # Publish user task to Redis for video overlay
        await self._call_set_user_task_tool(user_message)

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        max_iterations = self.config.llm.max_iterations
        iteration = 0
        planning_phase_complete = False

        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"--- Iteration {iteration}/{max_iterations} ---")

            # Prepare messages for LLM
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

            # Call LLM API
            try:
                # For Ollama, handle differently (no function calling)
                if self.llm_client.api_choice == "ollama":
                    self.logger.info("Using Ollama (text-based mode)")

                    response_text = self.llm_client.chat_completion(messages)

                    # Check if this is planning phase
                    is_planning, planning_text = self._extract_planning_phase(response_text)

                    if is_planning and not planning_phase_complete:
                        self.logger.info("=" * 80)
                        self.logger.info("CHAIN-OF-THOUGHT: PLANNING PHASE")
                        self.logger.info(planning_text)
                        self.logger.info("=" * 80)

                        print("\n" + "=" * 70)
                        print("💭 CHAIN-OF-THOUGHT REASONING")
                        print("=" * 70)
                        print(planning_text)
                        print("=" * 70 + "\n")

                        planning_phase_complete = True

                    # Add to history and return
                    self.conversation_history.append({"role": "assistant", "content": response_text})

                    self.logger.info(f"Assistant response: {response_text}")
                    self.logger.info("=" * 80)

                    return response_text

                else:
                    # OpenAI, Groq, Gemini - all support function calling
                    self.logger.info(f"Using {self.llm_client.api_choice.upper()} with function calling")

                    tools_formatted = self._convert_tools_to_function_format()

                    # PHASE 1: FORCE PLANNING - Don't provide tools on first call
                    if not planning_phase_complete and self.config.llm.require_planning:
                        self.logger.info("PHASE 1: Requesting planning (tools disabled)")

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
                            self.logger.info("=" * 80)
                            self.logger.info("CHAIN-OF-THOUGHT: PLANNING PHASE")
                            self.logger.info(planning_text)
                            self.logger.info("=" * 80)

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

                            self.logger.info("Planning phase complete. Proceeding to execution with function calling...")

                            # Continue to next iteration for tool execution
                            continue

                    # PHASE 2: EXECUTION - Now allow tool calls
                    self.logger.info("PHASE 2: Execution (tools enabled)")

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
                        self.logger.info(f"LLM requested {len(assistant_message.tool_calls)} tool call(s)")

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

                        self.logger.info(f"Final assistant response: {final_response}")
                        self.logger.info("=" * 80)

                        # Add to history
                        self.conversation_history.append({"role": "assistant", "content": final_response})

                        return final_response

            except Exception as e:
                error_msg = f"Error calling LLM API: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.logger.info("=" * 80)

                print(f"✗ {error_msg}")
                return error_msg

        self.logger.warning("Maximum iterations reached")
        self.logger.info("=" * 80)
        return "Maximum iterations reached. Task may be incomplete."

    def print_available_tools(self):
        """Print all available tools."""
        self.logger.info("Listing available tools")

        print("\n📋 Available Tools:")
        print("=" * 60)
        for tool in self.available_tools:
            print(f"\n🔧 {tool.name}")
            print(f"   {tool.description[:80]}...")
        print("=" * 60 + "\n")

    async def interactive_mode(self):
        """Run interactive chat mode."""
        self.logger.info("=" * 60)
        self.logger.info("INTERACTIVE MODE STARTED")
        self.logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("🤖 ROBOT CONTROL ASSISTANT (Universal LLM + CoT + Config)")
        print("=" * 60)
        print(f"\nUsing: {self.llm_client.api_choice.upper()} - {self.llm_client.llm}")
        print(f"Log file: {self.logger.handlers[0].baseFilename}")
        print(f"Config: {self.config.robot.type} robot, simulation={self.config.robot.simulation}")

        if self.text_overlay_manager:
            print(f"Video overlays: Enabled (Redis: {self.config.redis.host}:{self.config.redis.port})")
        else:
            print("Video overlays: Disabled")

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

                self.logger.info(f"User input: {user_input}")

                if user_input.lower() in ["quit", "exit", "q"]:
                    self.logger.info("User requested exit")
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "tools":
                    self.print_available_tools()
                    continue

                if user_input.lower() == "clear":
                    self.conversation_history = []
                    self.logger.info("Conversation history cleared")
                    print("✓ Conversation history cleared.\n")
                    continue

                if user_input.lower() == "switch":
                    self.logger.info("User requested provider switch")

                    print("\n🔄 Current provider:", self.llm_client.api_choice.upper())
                    print("Available: openai, groq, gemini, ollama")
                    new_api = input("Switch to (or press Enter to cancel): ").strip().lower()

                    if new_api in ["openai", "groq", "gemini", "ollama"]:
                        try:
                            # Get model for new provider from config
                            new_model = None
                            if new_api in self.config.llm.providers:
                                new_model = self.config.llm.providers[new_api].default_model

                            self.llm_client = LLMClient(
                                api_choice=new_api,
                                llm=new_model,
                                temperature=self.config.llm.temperature,
                                max_tokens=self.config.llm.max_tokens,
                            )
                            self.logger.info(f"Switched to {new_api.upper()} - {self.llm_client.llm}")
                            print(f"✓ Switched to {new_api.upper()} - {self.llm_client.llm}\n")
                        except Exception as e:
                            self.logger.error(f"Failed to switch provider: {e}", exc_info=True)
                            print(f"✗ Failed to switch: {e}\n")
                    continue

                print()  # Empty line for readability

                # Process the message (will show chain-of-thought reasoning and publish to Redis)
                response = await self.chat(user_input)

                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                self.logger.info("Interrupted by user (Ctrl+C)")
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}", exc_info=True)
                print(f"\n✗ Error: {e}\n")

            self.logger.info("=" * 60)
            self.logger.info("INTERACTIVE MODE ENDED")
            self.logger.info("=" * 60)

    async def run_command(self, command: str) -> str:
        """Run a single command and return response.

        Args:
            command: Natural language command

        Returns:
            Assistant's response
        """
        self.logger.info("=" * 60)
        self.logger.info("SINGLE COMMAND MODE")
        self.logger.info(f"  Command: {command}")
        self.logger.info("=" * 60)

        response = await self.chat(command)

        self.logger.info(f"Command completed. Response: {response}")
        self.logger.info("=" * 60)

        return response


# ============================================================================
# MAIN ENTRY POINT WITH CONFIG
# ============================================================================


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Universal MCP Client with Configuration Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python client/fastmcp_universal_client_with_config.py

  # Specify custom config
  python client/fastmcp_universal_client_with_config.py --config my_config.yaml

  # Override LLM provider
  python client/fastmcp_universal_client_with_config.py --api groq

  # Single command mode
  python client/fastmcp_universal_client_with_config.py --command "What objects do you see?"
        """,
    )

    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument("--environment", type=str, help="Environment name (dev, prod, test)")
    parser.add_argument("--api", choices=["openai", "groq", "gemini", "ollama"], help="Override LLM provider")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--command", help="Single command to execute")

    return parser.parse_args()


async def main():
    """Main entry point with configuration."""

    # Load environment variables
    load_dotenv(dotenv_path="secrets.env")

    # Parse arguments
    args = parse_arguments()

    try:
        # Load configuration
        print("📂 Loading configuration...")
        config = load_config(config_path=args.config, environment=args.environment or os.getenv("ROBOT_ENV"))
        print("✓ Configuration loaded successfully\n")

    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("MAIN ENTRY POINT")
    logger.info(f"  Arguments: {vars(args)}")
    logger.info("=" * 80)

    # Create client with config
    client = RobotUniversalMCPClient(
        config=config,
        api_choice=args.api,
        model=args.model,
    )

    try:
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
    finally:
        await client.disconnect()
        logger.info("=" * 80)
        logger.info("CLIENT SESSION ENDED")
        logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

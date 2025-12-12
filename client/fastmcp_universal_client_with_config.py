# client/fastmcp_universal_client_with_config.py
"""
Universal FastMCP Client with Configuration Management

Uses ConfigManager for centralized settings instead of scattered parameters.

Usage:
    # Use default config
    python client/fastmcp_universal_client_with_config.py

    # Specify config file
    python client/fastmcp_universal_client_with_config.py --config path/to/config.yaml

    # Override LLM provider
    python client/fastmcp_universal_client_with_config.py --api openai

    # Single command mode
    python client/fastmcp_universal_client_with_config.py --command "What objects do you see?"
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from llm_client import LLMClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager, load_config

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
# CLIENT CLASS WITH CONFIG
# ============================================================================


class RobotUniversalMCPClient:
    """Universal MCP Client with configuration management."""

    def __init__(
        self,
        config: ConfigManager,
        api_choice: str = None,
        model: str = None,
    ):
        """
        Initialize client with configuration.

        Args:
            config: ConfigManager instance
            api_choice: Override default LLM provider
            model: Override default model
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
1. {"ALWAYS start with PHASE 1 (Planning) before calling tools" if self.config.llm.require_planning else "Plan your actions"}
2. Always call get_detected_objects first before pick/place
3. Use exact coordinates from detected objects
4. Match object names EXACTLY as returned by detection
5. Use the FUNCTION CALLING API - never use XML tags
6. Provide clear feedback about actions
7. **MANDATORY FINAL STEP**: After completing ALL manipulation tasks, you MUST call move2observation_pose

Location options for placement:
- "left next to" - places left
- "right next to" - places right
- "above" - places above (farther in X)
- "below" - places below (closer in X)
- "on top of" - stacks on top
- "close to" - near coordinate

Always verify object positions before manipulation."""

        return prompt

    async def connect(self):
        """Connect to FastMCP server."""
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

    # [Rest of the client methods remain the same as original]
    # _convert_tools_to_function_format, call_tool, process_tool_calls,
    # chat, interactive_mode, etc.


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

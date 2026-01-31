# server/fastmcp_robot_server_communicative.py
"""
Enhanced FastMCP Robot Server with Communicative Explanations

This version adds LLM-generated explanations before tool execution,
making the robot more transparent and user-friendly.
"""

import argparse
import functools
import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Union

from fastmcp import FastMCP
from pydantic import ValidationError
from robot_environment import Environment
from robot_workspace import Location

from .schemas import (
    GetDetectedObjectsInput,
    PickPlaceInput,
)

# Import LLM client for explanations
try:
    from llm_client import LLMClient

    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False
    print("⚠️ llm_client not available - explanations disabled")

# ============================================================================
# LOGGING SETUP
# ============================================================================

os.makedirs("log", exist_ok=True)
log_filename = os.path.join("log", f'mcp_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
    force=True,
)

logger = logging.getLogger("FastMCPRobotServer")
logging.getLogger("RobotUniversalMCPClient").setLevel(logging.INFO)
logging.getLogger("robot_environment").setLevel(logging.INFO)

LOG_FILE = log_filename

logger.info("=" * 80)
logger.info(f"Communicative MCP Robot Server starting - Log file: {log_filename}")
logger.info("=" * 80)

# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================


class ExplanationGenerator:
    """Generates natural language explanations for tool calls using an LLM."""

    # Tool importance levels - determines which get voice output
    VOICE_PRIORITY = {
        # Critical operations - always speak
        "pick_place_object": "high",
        "pick_object": "high",
        "place_object": "high",
        "push_object": "high",
        "calibrate": "high",
        # Important operations - speak sometimes
        "move2observation_pose": "medium",
        "get_detected_objects": "medium",
        "get_largest_free_space_with_center": "medium",
        # Low priority - rarely speak
        "get_detected_object": "low",
        "get_largest_detected_object": "low",
        "get_smallest_detected_object": "low",
        "get_workspace_coordinate_from_point": "low",
        "get_object_labels_as_string": "low",
        "add_object_name2object_labels": "low",
        "speak": "low",  # Don't announce speak operations
    }

    def __init__(self, api_choice: str = "groq", model: str = None, verbose: bool = False):
        """Initialize the explanation generator.

        Args:
            api_choice: LLM provider (groq, openai, gemini, ollama)
            model: Specific model name (None = use default)
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.enabled = HAS_LLM_CLIENT

        if not self.enabled:
            logger.warning("LLM client not available - explanations disabled")
            return

        try:
            self.llm_client = LLMClient(
                api_choice=api_choice, llm=model, temperature=0.7, max_tokens=150  # Keep explanations concise
            )
            logger.info(f"Explanation generator initialized: {api_choice} - {self.llm_client.llm}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.enabled = False

    def should_speak(self, tool_name: str, force: bool = False) -> bool:
        """Determine if this tool call should generate speech.

        Args:
            tool_name: Name of the tool being called
            force: If True, always speak regardless of priority

        Returns:
            bool: True if should use TTS
        """
        if force:
            return True

        priority = self.VOICE_PRIORITY.get(tool_name, "low")

        # High priority always speaks
        if priority == "high":
            return True

        # Medium priority speaks 50% of the time (could be configurable)
        if priority == "medium":
            import random

            return random.random() > 0.5

        # Low priority rarely speaks (10% of the time)
        if priority == "low":
            import random

            return random.random() > 0.9

        return False

    def generate_explanation(self, tool_name: str, tool_description: str, arguments: dict, context: str = "") -> str:
        """Generate a natural language explanation for a tool call.

        Args:
            tool_name: Name of the tool being called
            tool_description: Tool's description from MCP
            arguments: Tool arguments
            context: Additional context about the current state

        Returns:
            str: Natural language explanation
        """
        if not self.enabled:
            return f"Calling {tool_name}"

        try:
            # Build prompt for explanation
            prompt = self._build_explanation_prompt(tool_name, tool_description, arguments, context)

            # Generate explanation using LLM
            explanation = self.llm_client.chat_completion(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful robot assistant explaining your actions to users in a friendly, concise way. Keep explanations to 1-2 sentences.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            # Clean up explanation
            explanation = explanation.strip()

            # Add emoji for personality (optional)
            if tool_name in ["pick_place_object", "pick_object", "place_object"]:
                explanation = "🤖 " + explanation
            elif tool_name == "move2observation_pose":
                explanation = "👁️ " + explanation
            elif "detect" in tool_name.lower():
                explanation = "🔍 " + explanation

            return explanation

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            # Fallback to simple explanation
            return self._generate_fallback_explanation(tool_name, arguments)

    def _build_explanation_prompt(self, tool_name: str, tool_description: str, arguments: dict, context: str) -> str:
        """Build the prompt for explanation generation."""

        # Format arguments nicely
        args_str = ", ".join([f"{k}={v}" for k, v in arguments.items()])

        prompt = f"""Explain in 1-2 friendly sentences what the robot is about to do:

Tool: {tool_name}
Description: {tool_description}
Arguments: {args_str}
{f"Context: {context}" if context else ""}

Generate a natural explanation that a user would understand. Be concise and friendly.
Examples:
- "I'm moving to observe the workspace so I can see all the objects clearly."
- "Let me pick up the pencil from its current position."
- "I'll place the cube right next to the red object you specified."
- "Checking what objects are currently in my workspace."

Your explanation:"""

        return prompt

    def _generate_fallback_explanation(self, tool_name: str, arguments: dict) -> str:
        """Generate a simple fallback explanation without LLM."""

        # Simple templates based on tool name
        templates = {
            "pick_place_object": lambda a: f"Picking up {a.get('object_name', 'object')} and placing it at the target location",
            "pick_object": lambda a: f"Picking up {a.get('object_name', 'object')}",
            "place_object": lambda a: "Placing the object I'm holding",
            "push_object": lambda a: f"Pushing {a.get('object_name', 'object')} {a.get('direction', 'forward')}",
            "move2observation_pose": lambda a: "Moving to observation position to see the workspace",
            "get_detected_objects": lambda a: "Scanning the workspace for objects",
            "calibrate": lambda a: "Calibrating my joints for accurate movement",
        }

        if tool_name in templates:
            return templates[tool_name](arguments)

        return f"Executing {tool_name.replace('_', ' ')}"


# Global explanation generator instance
explanation_generator = None


def init_explanation_generator(api_choice: str = "groq", verbose: bool = False):
    """Initialize the global explanation generator."""
    global explanation_generator
    explanation_generator = ExplanationGenerator(api_choice=api_choice, verbose=verbose)


# ============================================================================
# ENHANCED LOGGING DECORATOR
# ============================================================================


def log_tool_call_with_explanation(func):
    """Decorator to log tool calls and generate explanations."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__

        # Log incoming call
        logger.info("-" * 60)
        logger.info(f"TOOL CALL: {tool_name}")

        if args:
            logger.info(f"  Args: {args}")
        if kwargs:
            logger.info(f"  Kwargs: {kwargs}")

        # Generate explanation
        explanation = ""
        if explanation_generator and explanation_generator.enabled:
            try:
                # Get tool description from docstring
                tool_description = func.__doc__.split("\n")[0] if func.__doc__ else ""

                # Generate explanation
                explanation = explanation_generator.generate_explanation(
                    tool_name=tool_name, tool_description=tool_description, arguments=kwargs, context=""
                )

                logger.info(f"  Explanation: {explanation}")

                # Speak explanation if priority is high enough
                should_speak = explanation_generator.should_speak(tool_name)

                if should_speak and env is not None:
                    # Use text-to-speech for important operations
                    try:
                        env.oralcom_call_text2speech_async(explanation)
                    except Exception as e:
                        logger.warning(f"TTS failed: {e}")

            except Exception as e:
                logger.error(f"Failed to generate explanation: {e}")

        try:
            # Execute tool
            result = func(*args, **kwargs)

            # Log result
            logger.info(f"  Result: {result}")
            logger.info("  Status: SUCCESS")

            return result

        except Exception as e:
            # Log error
            logger.error(f"  Error: {str(e)}", exc_info=True)
            logger.info("  Status: FAILED")
            raise
        finally:
            logger.info("-" * 60)

    return wrapper


# ============================================================================
# VALIDATION DECORATOR (unchanged)
# ============================================================================


def validate_input(model_class):
    """Decorator to validate tool inputs using Pydantic models."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                validated_data = model_class(**kwargs)
                result = func(*args, **validated_data.model_dump())
                return result
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error["loc"])
                    msg = error["msg"]
                    errors.append(f"{field}: {msg}")

                error_msg = f"❌ Validation Error in {func.__name__}:\n" + "\n".join(f"  • {err}" for err in errors)
                logger.error(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"❌ Error in {func.__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg

        return wrapper

    return decorator


# ============================================================================
# MCP SETUP
# ============================================================================

mcp = FastMCP("robot-environment")
env = None
robot = None


def initialize_environment(el_api_key="", use_simulation=True, robot_id="niryo", verbose=False, start_camera_thread=False):
    """Initialize the robot environment with given parameters."""
    global env, robot

    logger.info("=" * 60)
    logger.info("ENVIRONMENT INITIALIZATION")
    logger.info(f"  Robot ID: {robot_id}")
    logger.info(f"  Simulation: {use_simulation}")
    logger.info(f"  Verbose: {verbose}")
    logger.info(f"  Camera Thread: {start_camera_thread}")
    logger.info("=" * 60)

    env = Environment(
        el_api_key=el_api_key,
        use_simulation=use_simulation,
        robot_id=robot_id,
        verbose=verbose,
        start_camera_thread=start_camera_thread,
    )
    robot = env.robot()

    logger.info("Environment initialized successfully")
    logger.info("=" * 60)


# ============================================================================
# EXAMPLE ENHANCED TOOLS (showing pattern)
# ============================================================================


@mcp.tool
@log_tool_call_with_explanation
@validate_input(PickPlaceInput)
def pick_place_object(
    object_name: str,
    pick_coordinate: List[float],
    place_coordinate: List[float],
    location: Union[Location, str, None] = None,
    z_offset: float = 0.001,
) -> str:
    """
    Complete pick-and-place operation in a single call.

    Args:
        object_name: Object label
        pick_coordinate: World coordinates [x, y] in meters
        place_coordinate: Target coordinates [x, y] in meters
        location: Relative placement position
        z_offset: Additional height offset in meters

    Returns:
        str: Success message or error description
    """
    try:
        success = robot.pick_place_object(
            object_name=object_name,
            pick_coordinate=pick_coordinate,
            place_coordinate=place_coordinate,
            location=location,
            z_offset=z_offset,
        )

        if success:
            location_str = f" {location} coordinate" if location else " at"
            z_offset_str = f" (z_offset: {z_offset:.3f}m)" if z_offset != 0.001 else ""
            return (
                f"✓ Successfully picked '{object_name}' from [{pick_coordinate[0]:.3f}, {pick_coordinate[1]:.3f}]{z_offset_str} "
                f"and placed it{location_str} [{place_coordinate[0]:.3f}, {place_coordinate[1]:.3f}]"
            )
        else:
            return f"❌ Failed to pick and place '{object_name}'"
    except Exception as e:
        return f"❌ Error during pick_place_object: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def move2observation_pose(workspace_id: str) -> str:
    """
    Move robot to observation position above workspace.

    Args:
        workspace_id: ID of the workspace

    Returns:
        str: Success message or error description
    """
    try:
        if not workspace_id or not isinstance(workspace_id, str):
            return "❌ Validation Error: workspace_id must be a non-empty string"

        robot.move2observation_pose(workspace_id)
        return f"✓ Moved to observation pose for workspace '{workspace_id}'"
    except Exception as e:
        return f"❌ Error moving to observation pose: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
@validate_input(GetDetectedObjectsInput)
def get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: Optional[List[float]] = None,
    label: Optional[str] = None,
) -> str:
    """
    Get list of detected objects with optional filters.

    Args:
        location: Spatial filter
        coordinate: Reference coordinate [x, y]
        label: Filter by object label

    Returns:
        str: JSON string of detected objects
    """
    try:
        env.robot_move2home_observation_pose()
        time.sleep(1)

        detected_objects = env.get_detected_objects()
        objects = detected_objects.get_detected_objects_serializable(location, coordinate, label)

        if objects:
            import json

            return f"✓ Found {len(objects)} object(s):\n{json.dumps(objects, indent=2)}"
        else:
            return "✓ No objects detected matching the criteria"
    except Exception as e:
        return f"❌ Error getting detected objects: {str(e)}"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point when running as script."""
    parser = argparse.ArgumentParser(description="Communicative FastMCP Robot Server with LLM Explanations")
    parser.add_argument("--robot", choices=["niryo", "widowx"], default="niryo")
    parser.add_argument("--no-simulation", action="store_false", dest="simulation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--explanation-api",
        default="groq",
        choices=["openai", "groq", "gemini", "ollama"],
        help="LLM provider for generating explanations",
    )
    parser.add_argument("--no-explanations", action="store_true", help="Disable LLM-generated explanations")

    args = parser.parse_args()

    # Log startup configuration
    logger.info("=" * 80)
    logger.info("SERVER CONFIGURATION")
    logger.info(f"  Robot:        {args.robot}")
    logger.info(f"  Simulation:   {args.simulation}")
    logger.info(f"  Host:         {args.host}")
    logger.info(f"  Port:         {args.port}")
    logger.info(f"  Camera:       {not args.no_camera}")
    logger.info(f"  Verbose:      {args.verbose}")
    logger.info(f"  Explanations: {'Disabled' if args.no_explanations else f'Enabled ({args.explanation_api})'}")
    logger.info(f"  Log File:     {log_filename}")
    logger.info("=" * 80)

    # Print to console
    print("=" * 60)
    print("COMMUNICATIVE FASTMCP ROBOT SERVER")
    print("=" * 60)
    print(f"Robot:        {args.robot}")
    print(f"Simulation:   {args.simulation}")
    print(f"Host:         {args.host}")
    print(f"Port:         {args.port}")
    print(f"Camera:       {not args.no_camera}")
    print(f"Explanations: {'Disabled' if args.no_explanations else f'Enabled ({args.explanation_api})'}")
    print(f"Log File:     {log_filename}")
    print("=" * 60)
    print(f"\nServer running at: http://{args.host}:{args.port}")
    print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    print(f"\nMonitor logs: tail -f {log_filename}")
    print("\n🤖 Enhanced with LLM-generated explanations")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Initialize environment
    initialize_environment(
        el_api_key="",
        use_simulation=args.simulation,
        robot_id=args.robot,
        verbose=args.verbose,
        start_camera_thread=not args.no_camera,
    )

    # Initialize explanation generator
    if not args.no_explanations:
        init_explanation_generator(api_choice=args.explanation_api, verbose=args.verbose)
        logger.info("Explanation generator initialized")
    else:
        logger.info("Explanations disabled by user")

    logger.info("Starting MCP server with explanations...")

    # Run server
    try:
        mcp.run(transport="sse", host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        print("\n\nShutting down server...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("=" * 80)
        logger.info("SERVER STOPPED")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

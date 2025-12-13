# server/fastmcp_robot_server_unified.py
"""
Complete Unified FastMCP Robot Server
Combines features from:
- Basic server (validation, all tools)
- Communicative server (LLM explanations)
- Config-based server (centralized configuration)
- Text overlay support (video recording integration)
"""

import argparse
import functools
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import ValidationError
from robot_environment import Environment
from robot_workspace import Location

# Import schemas
sys.path.insert(0, str(Path(__file__).parent))
from schemas import (
    GetDetectedObjectsInput,
    PickObjectInput,
    PickPlaceInput,
    PlaceObjectInput,
    PushObjectInput,
    WorkspacePointInput,
)

# Try importing optional dependencies
HAS_LLM_CLIENT = False
HAS_CONFIG = False
HAS_TEXT_OVERLAY = False

try:
    from llm_client import LLMClient

    HAS_LLM_CLIENT = True
except ImportError:
    print("⚠️ llm_client not available - explanations disabled")

try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.config_manager import load_config

    HAS_CONFIG = True
except ImportError:
    print("⚠️ config_manager not available - using command-line args")

try:
    from redis_robot_comm import RedisTextOverlayManager

    HAS_TEXT_OVERLAY = True
except ImportError:
    print("⚠️ redis_robot_comm not available - video text overlays disabled")

# ============================================================================
# LOGGING SETUP
# ============================================================================

os.makedirs("log", exist_ok=True)
log_filename = os.path.join("log", f'mcp_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, encoding="utf-8")],
    force=True,
)

logger = logging.getLogger("FastMCPRobotServer")
logging.getLogger("robot_environment").setLevel(logging.INFO)

logger.info("=" * 80)
logger.info(f"Unified MCP Robot Server starting - Log file: {log_filename}")
logger.info("=" * 80)


# ============================================================================
# EXPLANATION GENERATOR (from communicative server)
# ============================================================================


class ExplanationGenerator:
    """Generates natural language explanations for tool calls."""

    VOICE_PRIORITY = {
        "pick_place_object": "high",
        "pick_object": "high",
        "place_object": "high",
        "push_object": "high",
        "move2by": "high",
        "calibrate": "high",
        "move2observation_pose": "medium",
        "get_detected_objects": "medium",
        "get_largest_free_space_with_center": "medium",
        "get_detected_object": "low",
        "get_largest_detected_object": "low",
        "get_smallest_detected_object": "low",
        "get_detected_objects_sorted": "low",
        "get_workspace_coordinate_from_point": "low",
        "get_object_labels_as_string": "low",
        "add_object_name2object_labels": "low",
        "clear_collision_detected": "low",
        "speak": "never",
    }

    def __init__(self, api_choice: str = "groq", model: str = None, verbose: bool = False):
        self.verbose = verbose
        self.enabled = HAS_LLM_CLIENT

        if not self.enabled:
            logger.warning("LLM client not available - explanations disabled")
            return

        try:
            self.llm_client = LLMClient(api_choice=api_choice, llm=model, temperature=0.7, max_tokens=150)
            logger.info(f"Explanation generator initialized: {api_choice} - {self.llm_client.llm}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.enabled = False

    def should_speak(self, tool_name: str) -> bool:
        """Determine if this tool call should generate speech."""
        import random

        priority = self.VOICE_PRIORITY.get(tool_name, "low")

        if priority == "high":
            return True
        elif priority == "medium":
            return random.random() > 0.5
        elif priority == "low":
            return random.random() > 0.9
        return False

    def generate_explanation(self, tool_name: str, tool_description: str, arguments: dict) -> str:
        """Generate a natural language explanation for a tool call."""
        if not self.enabled:
            return self._generate_fallback_explanation(tool_name, arguments)

        try:
            prompt = self._build_explanation_prompt(tool_name, tool_description, arguments)

            explanation = self.llm_client.chat_completion(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful robot assistant explaining your actions to users in a friendly, concise way. Keep explanations to 1-2 sentences.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            explanation = explanation.strip()

            # Add emoji
            if tool_name in ["pick_place_object", "pick_object", "place_object", "move2by"]:
                explanation = "🤖 " + explanation
            elif tool_name == "move2observation_pose":
                explanation = "👁️ " + explanation
            elif "detect" in tool_name.lower():
                explanation = "🔍 " + explanation

            return explanation

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return self._generate_fallback_explanation(tool_name, arguments)

    def _build_explanation_prompt(self, tool_name: str, tool_description: str, arguments: dict) -> str:
        args_str = ", ".join([f"{k}={v}" for k, v in arguments.items()])

        prompt = f"""Explain in 1-2 friendly sentences what the robot is about to do:

Tool: {tool_name}
Description: {tool_description}
Arguments: {args_str}

Generate a natural explanation that a user would understand. Be concise and friendly.
Examples:
- "I'm moving to observe the workspace so I can see all the objects clearly."
- "Let me pick up the pencil from its current position."
- "I'll place the cube right next to the red object you specified."

Your explanation:"""
        return prompt

    def _generate_fallback_explanation(self, tool_name: str, arguments: dict) -> str:
        """Generate a simple fallback explanation without LLM."""
        templates = {
            "pick_place_object": lambda a: f"Picking up {a.get('object_name', 'object')} and placing it at the target location",
            "pick_object": lambda a: f"Picking up {a.get('object_name', 'object')}",
            "place_object": lambda a: "Placing the object I'm holding",
            "push_object": lambda a: f"Pushing {a.get('object_name', 'object')} {a.get('direction', 'forward')}",
            "move2by": lambda a: f"Moving {a.get('object_name', 'object')} {a.get('distance', 0)} meters {a.get('direction', '')}",
            "move2observation_pose": lambda a: "Moving to observation position to see the workspace",
            "get_detected_objects": lambda a: "Scanning the workspace for objects",
            "calibrate": lambda a: "Calibrating my joints for accurate movement",
        }

        if tool_name in templates:
            return templates[tool_name](arguments)
        return f"Executing {tool_name.replace('_', ' ')}"


# ============================================================================
# GLOBAL STATE
# ============================================================================

mcp = FastMCP("robot-environment")
env = None
robot = None
config = None
explanation_generator = None
text_overlay_manager = None
current_user_task = None


# ============================================================================
# DECORATORS
# ============================================================================


def validate_input(model_class):
    """Decorator to validate tool inputs using Pydantic models."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Validate input using Pydantic model
                validated_data = model_class(**kwargs)
                # Call original function with validated data
                result = func(*args, **validated_data.model_dump())
                return result
            except ValidationError as e:
                # Format validation errors nicely
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


def log_tool_call(func):
    """Decorator to log all tool calls with parameters and results."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__

        # Log incoming call
        logger.info("-" * 60)
        logger.info(f"TOOL CALL: {tool_name}")

        # Log arguments (be careful with sensitive data)
        if args:
            logger.info(f"  Args: {args}")
        if kwargs:
            logger.info(f"  Kwargs: {kwargs}")

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


def log_tool_call_with_explanation(func):
    """Decorator to log tool calls and generate explanations."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__

        # Log incoming call
        logger.info("-" * 60)
        logger.info(f"TOOL CALL: {tool_name}")

        # Log arguments (be careful with sensitive data)
        if args:
            logger.info(f"  Args: {args}")
        if kwargs:
            logger.info(f"  Kwargs: {kwargs}")

        # Generate explanation
        explanation = ""
        if explanation_generator and explanation_generator.enabled:
            try:
                tool_description = func.__doc__.split("\n")[0] if func.__doc__ else ""
                explanation = explanation_generator.generate_explanation(
                    tool_name=tool_name, tool_description=tool_description, arguments=kwargs
                )

                logger.info(f"  Explanation: {explanation}")

                # Publish to Redis for video overlay
                if text_overlay_manager:
                    text_overlay_manager.publish_robot_speech(
                        speech=explanation, duration_seconds=4.0, metadata={"tool_name": tool_name}
                    )

                # Speak if priority is high enough
                if explanation_generator.should_speak(tool_name) and env is not None:
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
# INITIALIZATION
# ============================================================================


def initialize_environment(el_api_key="", use_simulation=True, robot_id="niryo", verbose=False, start_camera_thread=False):
    """Initialize the robot environment."""
    global env, robot, text_overlay_manager

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

    # Initialize text overlay manager
    if HAS_TEXT_OVERLAY:
        try:
            text_overlay_manager = RedisTextOverlayManager()
            logger.info("Text overlay manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize text overlay manager: {e}")
            text_overlay_manager = None

    logger.info("Environment initialized successfully")
    logger.info("=" * 60)


# ============================================================================
# ENVIRONMENT TOOLS
# ============================================================================


@mcp.tool
@log_tool_call_with_explanation
def get_largest_free_space_with_center() -> str:
    """
    Determines the largest free space in the workspace in square metres and its center coordinate in metres.
    This method can be used to determine at which location an object can be placed safely.

    Example call:
    To pick a 'chocolate bar' and place it at the center of the largest free space of the workspace, call:

    largest_free_area_m2, center_x, center_y = agent.get_largest_free_space_with_center()

    robot.pick_place_object(
        object_name='chocolate bar',
        pick_coordinate=[-0.1, 0.01],
        place_coordinate=[center_x, center_y],
        location=Location.RIGHT_NEXT_TO
    )

    Returns:
        str: Description of the largest free space with area and center coordinates.
    """
    try:
        area_m2, center_x, center_y = env.get_largest_free_space_with_center()
        return f"✓ Largest free space: {area_m2:.4f} m² at center coordinates [{center_x:.3f}, {center_y:.3f}]"
    except Exception as e:
        return f"❌ Error getting largest free space: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
@validate_input(WorkspacePointInput)
def get_workspace_coordinate_from_point(workspace_id: str, point: str) -> str:
    """
    Get the world coordinate of a special point of the given workspace.

    Args:
        workspace_id (str): ID of workspace.
        point (str): description of point. Possible values are:
        - 'upper left corner': Returns the world coordinate of the upper left corner of the workspace.
        - 'upper right corner': Returns the world coordinate of the upper right corner of the workspace.
        - 'lower left corner': Returns the world coordinate of the lower left corner of the workspace.
        - 'lower right corner': Returns the world coordinate of the lower right corner of the workspace.
        - 'center point': Returns the world coordinate of the center of the workspace.

    Returns:
        str: (x,y) world coordinate of the point on the workspace that was specified by the argument point.
    """
    try:
        coord = env.get_workspace_coordinate_from_point(workspace_id, point)
        if coord:
            return f"✓ Coordinate of '{point}' in workspace '{workspace_id}': [{coord[0]:.3f}, {coord[1]:.3f}]"
        else:
            return f"❌ Could not get coordinate for '{point}' in workspace '{workspace_id}'"
    except Exception as e:
        return f"❌ Error getting workspace coordinate: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def get_object_labels_as_string() -> str:
    """
    Returns all object labels that the object detection model is able to detect as a comma separated string.
    Call this method if the user wants to know which objects the robot can pick or is able to detect.

    Returns:
        str: Comma-separated list of detectable objects. "chocolate bar, blue box, cigarette, ..."
    """
    try:
        labels = env.get_object_labels_as_string()
        return f"✓ Detectable objects: {labels}"
    except Exception as e:
        return f"❌ Error getting object labels: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def add_object_name2object_labels(object_name: str) -> str:
    """
    Call this method if the user wants to add another object to the list of recognizable objects. Adds the
    object called object_name to the list of recognizable objects.

    Args:
        object_name (str): The name of the object that should also be recognizable by the robot.

    Returns:
        str: Message saying that the given object_name was added to the list of recognizable objects.
    """
    try:
        if not object_name or not isinstance(object_name, str):
            return "❌ Validation Error: object_name must be a non-empty string"

        result = env.add_object_name2object_labels(object_name)
        return f"✓ {result}"
    except Exception as e:
        return f"❌ Error adding object name: {str(e)}"


# ============================================================================
# ROBOT TOOLS
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
    Command the pick-and-place robot arm to pick a specific object and place it using its gripper.
    The gripper will move to the specified 'pick_coordinate' and pick the named object. Then it will move to the
    specified 'place_coordinate' and place the object there. If you have to pick-and-place an object, call this
    function and not pick_object() followed by place_object().

    Example calls:
    robot.pick_place_object(
        object_name='chocolate bar',
        pick_coordinate=[-0.1, 0.01],
        place_coordinate=[0.1, 0.11],
        location=Location.RIGHT_NEXT_TO
    )
    --> Picks the chocolate bar that is located at world coordinates [-0.1, 0.01] and places it right next to an
    object that exists at world coordinate [0.1, 0.11].

    robot.pick_place_object(
        object_name='cube',
        pick_coordinate=[0.2, 0.05],
        place_coordinate=[0.3, 0.1],
        location=Location.ON_TOP_OF,
        z_offset=0.02
    )
    --> Picks the cube with 2cm z-offset (useful if on top of another object).

    Args:
        object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
        the robot's workspace.
        pick_coordinate (List[float]): The world coordinates [x, y] where the object should be picked up. Use these
        coordinates to identify the object's exact position.
        place_coordinate (List[float]): The world coordinates [x, y] where the object should be placed at.
        location (Location or str): Specifies the relative placement position of the picked object in relation to an
        object being at the 'place_coordinate'. Possible values are defined in the `Location` Enum:
            - `Location.LEFT_NEXT_TO`: Left of the reference object.
            - `Location.RIGHT_NEXT_TO`: Right of the reference object.
            - `Location.ABOVE`: Above the reference object.
            - `Location.BELOW`: Below the reference object.
            - `Location.ON_TOP_OF`: On top of the reference object.
            - `Location.INSIDE`: Inside the reference object.
            - `Location.NONE`: No specific location relative to another object.
        or 'left next to', 'right next to', 'above', 'below', 'on top of', 'inside'
        z_offset (float): Additional height offset in meters to apply when picking (default: 0.001).
        Useful for picking objects that are stacked on top of other objects.

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
@validate_input(PickObjectInput)
def pick_object(object_name: str, pick_coordinate: List[float], z_offset: float = 0.001) -> str:
    """
    Command the pick-and-place robot arm to pick up a specific object using its gripper.
    The gripper will move to the specified 'pick_coordinate' and pick the named object.

    Example calls:
    robot.pick_object("pen", [0.01, -0.15])
    --> Picks the pen that is located at world coordinates [0.01, -0.15].

    robot.pick_object("pen", [0.01, -0.15], z_offset=0.02)
    --> Picks pen with 2cm offset above detected position (useful for stacked objects).

    Args:
        object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
        the robot's workspace.
        pick_coordinate (List[float]): The world coordinates [x, y] where the object should be picked up. Use these
        coordinates to identify the object's exact position.
        z_offset (float): Additional height offset in meters to apply when picking (default: 0.001).
        Useful for picking objects that are stacked on top of other objects.
    Returns:
        str: Success message or error description
    """
    try:
        success = robot.pick_object(object_name=object_name, pick_coordinate=pick_coordinate, z_offset=z_offset)

        if success:
            z_offset_str = f" with z_offset {z_offset:.3f}m" if z_offset != 0.001 else ""
            return f"✓ Successfully picked '{object_name}' from [{pick_coordinate[0]:.3f}, {pick_coordinate[1]:.3f}]{z_offset_str}"
        else:
            return f"❌ Failed to pick '{object_name}'"
    except Exception as e:
        return f"❌ Error during pick_object: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
@validate_input(PlaceObjectInput)
def place_object(place_coordinate: List[float], location: Union[Location, str, None] = None) -> str:
    """
    Instruct the pick-and-place robot arm to place a picked object at the specified 'place_coordinate'. The
    function moves the gripper to the specified 'place_coordinate' and calculates the exact placement position from
    the given 'location'. Before calling this function you have to call pick_object() to pick an object.

    Example call:
    robot.place_object([0.2, 0.0], "left next to")
    --> Places the gripped object left next to world coordinate [0.2, 0.0].

    Args:
        place_coordinate (List[float]): Target coordinates [x, y] in meters.
        location (str, optional): Relative placement position. Options:
            - 'left next to': Place to the left
            - 'right next to': Place to the right
            - 'above': Place above (farther in X)
            - 'below': Place below (closer in X)
            - 'on top of': Stack on top
            - 'inside': Place inside container
            - 'close to': Within 2cm radius
            - None: Use exact coordinate (default)

    Returns:
        str: Success message or error description
    """
    try:
        success = robot.place_object(place_coordinate=place_coordinate, location=location)

        if success:
            location_str = f" {location} coordinate" if location else " at"
            return f"✓ Successfully placed object{location_str} [{place_coordinate[0]:.3f}, {place_coordinate[1]:.3f}]"
        else:
            return "❌ Failed to place object"
    except Exception as e:
        return f"❌ Error during place_object: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
@validate_input(PushObjectInput)
def push_object(object_name: str, push_coordinate: List[float], direction: str, distance: float) -> str:
    """
    Direct the pick-and-place robot arm to push a specific object to a new position.
    This function should only be called if it is not possible to pick the object.
    An object cannot be picked if its shorter side is larger than the gripper (~5cm).

    Example call:
    robot.push_object("large box", [0.25, 0.05], "right", 50.0)
    --> Pushes the large box 50mm to the right from its current position.

    Args:
        object_name (str): The name of the object to be pushed.
        Ensure the name matches an object in the robot's environment.
        push_coordinate: The world coordinates [x, y] where the object to be pushed is located.
        These coordinates indicate the initial position of the object.
        direction (str): The direction in which the object should be pushed.
        Valid options are: "up", "down", "left", "right".
        distance: The distance (in millimeters) to push the object in the specified direction.
        Ensure the value is within the robot's operational range.
    Returns:
        str: Success message or error description
    """
    try:
        success = robot.push_object(object_name, push_coordinate, direction, distance)

        if success:
            return (
                f"✓ Successfully pushed '{object_name}' from [{push_coordinate[0]:.3f}, {push_coordinate[1]:.3f}] "
                f"{direction} by {distance:.1f}mm"
            )
        else:
            return f"❌ Failed to push '{object_name}'"
    except Exception as e:
        return f"❌ Error during push_object: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def move2by(
    object_name: str,
    pick_coordinate: List[float],
    direction: str,
    distance: float,
    z_offset: float = 0.001,
) -> str:
    """
    Command the robot to pick an object and move it a given distance in a specified direction.
    This is a convenience function that combines pick and place operations.

    Example calls:
    robot.move2by(
        object_name='pencil',
        pick_coordinate=[-0.11, 0.21],
        direction='left',
        distance=0.02
    )
    --> Picks pencil at [-0.11, 0.21] and moves it 2cm to the left to [-0.11, 0.23].

    robot.move2by(
        object_name='cube',
        pick_coordinate=[-0.11, 0.21],
        direction='up',
        distance=0.03,
        z_offset=0.02
    )
    --> Picks cube with 2cm z-offset and moves it 3cm upwards.

    Args:
        object_name (str): Name of object to move. Must match detection exactly.
        pick_coordinate (List[float]): Current world coordinates [x, y] in meters.
        direction (str): Direction to move. Options: 'left', 'right', 'up', 'down'.
        distance (float): Distance in meters to move the object.
        z_offset (float): Height offset in meters for picking (default: 0.001).
            Useful if object is on top of or blocking another object.

    Returns:
        str: Success message or error description
    """
    try:
        if not isinstance(pick_coordinate, list) or len(pick_coordinate) != 2:
            return "❌ Validation Error: pick_coordinate must be a list of 2 numeric values [x, y]"

        if direction not in ["left", "right", "up", "down"]:
            return "❌ Validation Error: direction must be one of: left, right, up, down"

        place_coordinate = pick_coordinate.copy()
        if direction == "left":
            place_coordinate[1] += distance
        elif direction == "right":
            place_coordinate[1] -= distance
        elif direction == "up":
            place_coordinate[0] += distance
        elif direction == "down":
            place_coordinate[0] -= distance

        success = robot.pick_place_object(
            object_name=object_name,
            pick_coordinate=pick_coordinate,
            place_coordinate=place_coordinate,
            location=None,
            z_offset=z_offset,
        )

        if success:
            z_offset_str = f" (z_offset: {z_offset:.3f}m)" if z_offset != 0.001 else ""
            return (
                f"✓ Successfully picked '{object_name}' from [{pick_coordinate[0]:.3f}, {pick_coordinate[1]:.3f}]{z_offset_str} "
                f"and moved it {distance * 100:.1f}cm {direction}wards to [{place_coordinate[0]:.3f}, {place_coordinate[1]:.3f}]"
            )
        else:
            return f"❌ Failed to move '{object_name}'"
    except Exception as e:
        return f"❌ Error during move2by: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def move2observation_pose(workspace_id: str) -> str:
    """
    Move robot to observation position above the specified workspace.
    The gripper will hover over the workspace for optimal object detection.
    Must be called before picking or placing objects in a workspace.

    Example call:
    robot.move2observation_pose("niryo_ws")
    --> Moves to observation pose above niryo_ws workspace.

    Args:
        workspace_id (str): ID of the workspace (e.g., "niryo_ws", "gazebo_1").

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
def clear_collision_detected() -> str:
    """
    Reset the internal "collision_detected" flag of the Niryo robot.
    Call this after a collision event to resume normal operation.
    Note: This is Niryo-specific functionality.

    Returns:
        str: Success message or error description
    """
    try:
        robot.robot().robot_ctrl().clear_collision_detected()
        return "✓ Collision detection flag cleared"
    except Exception as e:
        return f"❌ Error clearing collision flag: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def calibrate() -> str:
    """
    Calibrate the robot's joints for accurate movement.
    This should be called if the robot shows positioning errors or after power-up.

    Returns:
        str: Success message or error description
    """
    try:
        success = robot.calibrate()

        if success:
            return "✓ Robot calibration completed successfully"
        else:
            return "❌ Robot calibration failed"
    except Exception as e:
        return f"❌ Error during calibration: {str(e)}"


# ============================================================================
# MCP TOOLS - OBJECT DETECTION
# ============================================================================


@mcp.tool
@log_tool_call_with_explanation
@validate_input(GetDetectedObjectsInput)
def get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: Optional[List[float]] = None,
    label: Optional[str] = None,
) -> str:
    """
    Get list of all objects detected by the camera in the workspace.
    Supports optional spatial and label filtering.

    Example calls:
    get_detected_objects()
    --> Returns all detected objects.

    get_detected_objects(location="close to", coordinate=[0.2, 0.0])
    --> Returns objects within 2cm of [0.2, 0.0].

    get_detected_objects(label="pencil")
    --> Returns all objects labeled "pencil".

    get_detected_objects(location="left next to", coordinate=[0.20, 0.0], label="cube")
    --> Returns cubes to the left of [0.20, 0.0].

    Args:
        location (Location or str, optional): Spatial filter. Options:
            - "left next to": Objects left of coordinate
            - "right next to": Objects right of coordinate
            - "above": Objects above coordinate (farther in X)
            - "below": Objects below coordinate (closer in X)
            - "close to": Objects within 2cm radius
            - None: No spatial filter (default)
        coordinate (List[float], optional): Reference coordinate [x, y] in meters.
            Required if location is specified.
        label (str, optional): Filter by object name. Case-sensitive.

    Returns:
        str: JSON string of detected objects with positions and dimensions, or error message
    """
    try:
        env.robot_move2home_observation_pose()
        # wait for robot to reach observation pose
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


@mcp.tool
@log_tool_call_with_explanation
def get_detected_object(coordinate: List[float], label: Optional[str] = None) -> str:
    """
    Retrieves a detected object at or near a specified world coordinate, optionally filtering by label.

    This method checks for objects detected by the camera that are close to the specified coordinate (within
    2 centimeters). If multiple objects meet the criteria, the first object in the list is returned.

    Example calls:
    get_detected_object([0.18, -0.05])
    --> Finds any object at or near [0.18, -0.05].

    get_detected_object([0.18, -0.05], label="pen")
    --> Finds specifically a "pen" at that location.

    Args:
        coordinate (List[float]): A 2D coordinate in the world coordinate system [x, y].
            Only objects within a 2-centimeter radius of this coordinate are considered.
        label (Optional[str]): An optional filter for the object's label. If specified, only an object
            with the matching label is returned.

    Returns:
        str: JSON string of the found object or message if not found
    """
    try:
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            return "❌ Validation Error: coordinate must be a list of 2 numeric values [x, y]"

        if not all(isinstance(x, (int, float)) for x in coordinate):
            return "❌ Validation Error: coordinate values must be numeric"

        env.robot_move2home_observation_pose()
        # wait for robot to reach observation pose
        time.sleep(1)

        detected_objects = env.get_detected_objects()
        obj = detected_objects.get_detected_object(coordinate, label, True)

        if obj:
            import json

            return f"✓ Found object near [{coordinate[0]:.3f}, {coordinate[1]:.3f}]:\n{json.dumps(obj, indent=2)}"
        else:
            return f"✓ No object found near [{coordinate[0]:.3f}, {coordinate[1]:.3f}]"
    except Exception as e:
        return f"❌ Error getting detected object: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def get_largest_detected_object() -> str:
    """
    Returns the largest detected object based on its size in square meters.

    Returns:
        str: Largest object information or error message
    """
    try:
        env.robot_move2home_observation_pose()
        # wait for robot to reach observation pose
        time.sleep(1)

        detected_objects = env.get_detected_objects()
        obj, size = detected_objects.get_largest_detected_object(True)

        if obj:
            import json

            return f"✓ Largest object ({size:.6f} m²):\n{json.dumps(obj, indent=2)}"
        else:
            return "✓ No objects detected"
    except Exception as e:
        return f"❌ Error getting largest object: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def get_smallest_detected_object() -> str:
    """
    Returns the smallest detected object based on its size in square meters.

    Returns:
        str: Smallest object information or error message
    """
    try:
        env.robot_move2home_observation_pose()
        time.sleep(1)

        detected_objects = env.get_detected_objects()
        obj, size = detected_objects.get_smallest_detected_object(True)

        if obj:
            import json

            return f"✓ Smallest object ({size:.6f} m²):\n{json.dumps(obj, indent=2)}"
        else:
            return "✓ No objects detected"
    except Exception as e:
        return f"❌ Error getting smallest object: {str(e)}"


@mcp.tool
@log_tool_call_with_explanation
def get_detected_objects_sorted(ascending: bool = True) -> str:
    """
    Get detected objects sorted by size.

    Args:
        ascending: If True, sort smallest to largest; if False, largest to smallest

    Returns:
        str: Sorted list of objects or error message
    """
    try:
        if not isinstance(ascending, bool):
            return "❌ Validation Error: ascending must be a boolean (true/false)"

        env.robot_move2home_observation_pose()
        time.sleep(1)

        detected_objects = env.get_detected_objects()
        objects = detected_objects.get_detected_objects_sorted(ascending, True)

        if objects:
            import json

            order = "smallest to largest" if ascending else "largest to smallest"
            return f"✓ Found {len(objects)} object(s) sorted {order}:\n{json.dumps(objects, indent=2)}"
        else:
            return "✓ No objects detected"
    except Exception as e:
        return f"❌ Error getting sorted objects: {str(e)}"


# OBJECT TOOLS

# @mcp.tool
# def get_object_label(myobject: Object) -> str:
#     """
#     Returns the label of the object (e.g., "chocolate bar").
#
#     Args:
#         myobject (Object): object of type Object you want to know the label for.
#
#     Returns:
#         str: the label of the object.
#     """
#     return myobject.label()
#
#
# @mcp.tool
# def get_object_coordinate(myobject: Object) -> List[float]:
#     """
#     Returns (x,y) world coordinates of the center of mass of the object, measured in meters.
#     At these coordinates you can pick the object. The List contains two float values.
#
#     Args:
#         myobject (Object): object of type Object you want to know the label for.
#
#     Returns:
#         List[float]: x,y world coordinates of the object. At these coordinates you can pick the object.
#     """
#     return myobject.coordinate()


# ============================================================================
# FEEDBACK TOOLS
# ============================================================================


@mcp.tool
@log_tool_call
def speak(text: str) -> str:
    """Make the robot speak a message using text-to-speech."""
    try:
        if not text or not isinstance(text, str):
            return "❌ Validation Error: text must be a non-empty string"

        env.oralcom_call_text2speech_async(text)
        return f"✓ Speaking: '{text}'"
    except Exception as e:
        return f"❌ Error during text-to-speech: {str(e)}"


@mcp.tool
def set_user_task(task: str) -> str:
    """
    Set the current user task for display in video recordings.

    Args:
        task: The user's natural language task/command

    Returns:
        str: Confirmation message
    """
    global current_user_task

    current_user_task = task
    logger.info(f"User task set: {task}")

    # Publish to Redis for video overlay
    if text_overlay_manager:
        text_overlay_manager.publish_user_task(task=task, metadata={"timestamp": time.time()})

    return f"✓ User task set: {task}"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified FastMCP Robot Server", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument("--environment", type=str, choices=["development", "production", "testing"])

    # Robot settings
    parser.add_argument("--robot", choices=["niryo", "widowx"], default="niryo")
    parser.add_argument("--no-simulation", action="store_false", dest="simulation")

    # Server settings
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    # Features
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-explanations", action="store_true")
    parser.add_argument("--explanation-api", default="groq", choices=["openai", "groq", "gemini", "ollama"])

    return parser.parse_args()


# python server/fastmcp_robot_server.py
# für realen Roboter
# python server/fastmcp_robot_server.py --no-simulation
def main():
    """Main entry point."""
    global config, explanation_generator

    load_dotenv(dotenv_path="secrets.env")
    args = parse_arguments()

    # Try loading configuration if available
    if HAS_CONFIG:
        try:
            config = load_config(config_path=args.config, environment=args.environment or os.getenv("ROBOT_ENV"))
            logger.info("Configuration loaded successfully")

            # Use config values
            robot_type = config.robot.type
            simulation = config.robot.simulation
            camera = config.robot.enable_camera
            host = config.server.host
            port = config.server.port

        except Exception as e:
            logger.warning(f"Config loading failed, using CLI args: {e}")
            config = None

    # Fall back to command-line arguments
    if config is None:
        robot_type = args.robot
        simulation = args.simulation
        camera = not args.no_camera
        host = args.host
        port = args.port

    # Print startup info
    print("=" * 60)
    print("UNIFIED FASTMCP ROBOT SERVER")
    print("=" * 60)
    print(f"Robot:        {robot_type}")
    print(f"Simulation:   {simulation}")
    print(f"Host:         {host}:{port}")
    print(f"Camera:       {camera}")
    print(f"Explanations: {'Disabled' if args.no_explanations else f'Enabled ({args.explanation_api})'}")
    print(f"Text Overlay: {'Enabled' if HAS_TEXT_OVERLAY else 'Disabled'}")
    print(f"Log File:     {log_filename}")
    print("=" * 60)
    print(f"\nServer: http://{host}:{port}")
    print(f"SSE:    http://{host}:{port}/sse")
    print(f"\nMonitor: tail -f {log_filename}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Initialize environment
    initialize_environment(
        el_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        use_simulation=simulation,
        robot_id=robot_type,
        verbose=args.verbose,
        start_camera_thread=camera,
    )

    # Initialize explanation generator
    if not args.no_explanations:
        explanation_generator = ExplanationGenerator(api_choice=args.explanation_api, verbose=args.verbose)

    # Run server
    try:
        mcp.run(transport="sse", host=host, port=port)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        logger.info("=" * 80)
        logger.info("SERVER STOPPED")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

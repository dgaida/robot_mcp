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
from .schemas import (
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
        "pick_place_object": 8,
        "pick_object": 8,
        "place_object": 8,
        "push_object": 7,
        "move2by": 7,
        "calibrate": 7,
        "move2observation_pose": 5,
        "get_detected_objects": 5,
        "get_largest_free_space_with_center": 5,
        "get_detected_object": 2,
        "get_largest_detected_object": 2,
        "get_smallest_detected_object": 2,
        "get_detected_objects_sorted": 2,
        "get_workspace_coordinate_from_point": 2,
        "get_object_labels_as_string": 2,
        "add_object_name2object_labels": 2,
        "clear_collision_detected": 1,
        "speak": 0,  # Never announce
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

    def should_speak(self, tool_name: str) -> int:
        """
        ✅ MODIFIED: Return priority instead of boolean.

        Returns:
            int: Priority value (0 = don't speak, 1-10 = speak with priority)
        """
        return self.VOICE_PRIORITY.get(tool_name, 0)

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
                priority = explanation_generator.should_speak(tool_name)
                if priority > 0 and env is not None:
                    try:
                        # Queue with priority (non-blocking)
                        success = env.oralcom_call_text2speech_async(explanation, priority=priority)
                        if not success:
                            logger.warning("Failed to queue TTS message")
                    except Exception as e:
                        logger.warning(f"TTS queueing failed: {e}")

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
    Determine the largest free space in the workspace and its center coordinate.

    This method analyzes the workspace to find the largest contiguous empty area,
    which is useful for safely placing objects without collisions.

    Examples:
        get_largest_free_space_with_center()
        # Returns: "0.0045 m² at center [0.240, -0.030]"

        # Use for safe placement:
        # 1. result = get_largest_free_space_with_center()
        # 2. Parse to get center coordinates
        # 3. place_object(center_coords, None)

        # Example workflow:
        # area, x, y = parse_result()
        # pick_place_object("cube", [0.2, 0.1], [x, y], None)

    Returns:
        str: Description of the largest free space with area in square meters
            and center coordinates in meters [x, y].
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
    Get the world coordinate of a special point in the workspace.

    Returns coordinates for workspace corners and center point, useful for
    boundary-aware placement and workspace organization.

    Examples:
        get_workspace_coordinate_from_point("niryo_ws", "upper left corner")
        # Returns: "✓ Coordinate of 'upper left corner': [0.337, 0.087]"

        get_workspace_coordinate_from_point("niryo_ws", "center point")
        # Returns: "✓ Coordinate of 'center point': [0.250, 0.000]"

        get_workspace_coordinate_from_point("niryo_ws", "lower right corner")
        # Returns: "✓ Coordinate of 'lower right corner': [0.163, -0.087]"

        # Use for boundary placement:
        # upper_left = get_workspace_coordinate_from_point("niryo_ws", "upper left corner")
        # pick_place_object("cube", [0.2, 0.1], upper_left, None)

        # Organize in corners:
        # place_object(get_workspace_coordinate_from_point("niryo_ws", "upper right corner"))

    Args:
        workspace_id (str): ID of the workspace (e.g., "niryo_ws", "gazebo_1").
            Must match a configured workspace in the system.
        point (str): Description of the point to query. Valid options:
            - "upper left corner": Farthest and leftmost point
            - "upper right corner": Farthest and rightmost point
            - "lower left corner": Closest and leftmost point
            - "lower right corner": Closest and rightmost point
            - "center point": Center of the workspace

    Returns:
        str: World coordinates [x, y] in meters of the specified point,
            or error message if workspace_id or point is invalid.

    Note:
        For Niryo workspace:
        - Upper left: [0.337, 0.087] (far and left)
        - Lower right: [0.163, -0.087] (close and right)
        - Center: ~[0.250, 0.000]
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
    Return all object labels that the detection model can recognize.

    This function returns a comma-separated string of all object types that the
    vision system is configured to detect. Use this to verify which objects the
    robot can identify in general.

    Examples:
        get_object_labels_as_string()
        # Returns: "pencil, pen, cube, cylinder, chocolate bar, cigarette, ..."

        # List available objects to user:
        # labels = get_object_labels_as_string()
        # print(f"I can detect: {labels}")

    Returns:
        str: Comma-separated list of detectable object labels,
            e.g., "chocolate bar, blue box, cigarette, pen, pencil, cube"

    Note:
        Call this method when the user asks "What objects can you see?" or
        "What can you pick up?" to show detection capabilities.
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
    Add a new object type to the list of recognizable objects.

    Extends the detection system to recognize additional object types. The vision
    model will attempt to detect this object in subsequent scans.

    Examples:
        add_object_name2object_labels("screwdriver")
        # Returns: "✓ Added 'screwdriver' to the list of recognizable objects"
        # Now the robot will look for screwdrivers in the workspace

        add_object_name2object_labels("red ball")
        # Adds "red ball" to detection labels

        add_object_name2object_labels("smartphone")
        # Vision system will now attempt to detect smartphones

        # Workflow:
        # 1. add_object_name2object_labels("wrench")
        # 2. get_object_labels_as_string()  # Verify "wrench" is included
        # 3. get_detected_objects()  # System will now detect wrenches

    Args:
        object_name (str): Name of the new object to recognize. Should be descriptive
            and match the object's appearance. Case-sensitive for future detection.

    Returns:
        str: Confirmation message that the object was added to recognition list,
            or error if object_name is invalid.

    Note:
        The vision model's ability to detect the new object depends on its training.
        Well-known objects (tools, office supplies) are more likely to be detected.
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
    Pick an object and place it at a target location in a single operation.

    The gripper moves to pick_coordinate, grasps the object, moves to place_coordinate,
    and releases it. Preferred over separate pick_object() and place_object() calls.

    Examples:
        pick_place_object("chocolate bar", [-0.1, 0.01], [0.1, 0.11], "right next to")
        # Picks chocolate bar and places it right next to object at [0.1, 0.11]

        pick_place_object("cube", [0.2, 0.05], [0.3, 0.1], "on top of", z_offset=0.02)
        # Picks cube with 2cm z-offset and places it on top of target object

        pick_place_object("pen", [0.15, -0.05], [0.25, 0.0], None)
        # Picks pen and places at exact coordinates (no relative positioning)

    Args:
        object_name (str): Name of object to pick. Must match detection exactly.
        pick_coordinate (List[float]): World coordinates [x, y] in meters where
            the object is currently located. Get from get_detected_objects().
        place_coordinate (List[float]): Target world coordinates [x, y] in meters.
            Final position depends on the location parameter.
        location (Union[Location, str, None]): Relative placement position.
            Possible values are defined in the `Location` Enum:
            - `Location.LEFT_NEXT_TO`: Left of the reference object.
            - `Location.RIGHT_NEXT_TO`: Right of the reference object.
            - `Location.ABOVE`: Above the reference object.
            - `Location.BELOW`: Below the reference object.
            - `Location.ON_TOP_OF`: On top of the reference object.
            - `Location.INSIDE`: Inside the reference object.
            - `Location.NONE`: No specific location relative to another object.
            Options as str: "left next to", "right next to", "above", "below", "on top of",
            "inside", or None for exact coordinates.
        z_offset (float): Height offset in meters for picking (default: 0.001).
            Use 0.02 or higher if object is stacked on another object.

    Returns:
        str: Success message with pick and place coordinates, or error description.

    Note:
        Always call get_detected_objects() first to get current object positions.
        Object names are case-sensitive and must match detection exactly.
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
    Pick up a specific object using the robot gripper.

    The gripper moves to pick_coordinate, closes to grasp the object, and lifts it.
    Must be followed by place_object() to complete the operation.

    Examples:
        pick_object("pen", [0.01, -0.15])
        # Picks pen at world coordinates [0.01, -0.15]

        pick_object("cube", [0.20, 0.05])
        # Picks cube with default z_offset

        pick_object("pen", [0.01, -0.15], z_offset=0.02)
        # Picks pen with 2cm offset above detected position
        # Useful if pen is on top of another object

    Args:
        object_name (str): Name of the object to pick. Ensure this name matches
            an object visible in the robot's workspace. Case-sensitive.
        pick_coordinate (List[float]): World coordinates [x, y] in meters where
            the object is located. Get these from get_detected_objects().
        z_offset (float): Additional height offset in meters (default: 0.001).
            Increase if object is on top of another object (e.g., 0.02 for 2cm).

    Returns:
        str: Success message with coordinates, or error description if failed.

    Note:
        Must be followed by place_object() to complete pick-and-place operation.
        For complete operation in one call, use pick_place_object() instead.
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
    Place a previously picked object at the specified location.

    Moves the gripper to place_coordinate and releases the object. Must be preceded
    by pick_object(). The exact placement position is calculated based on the location parameter.

    Examples:
        place_object([0.2, 0.0], "left next to")
        # Places gripped object to the left of coordinate [0.2, 0.0]

        place_object([0.25, 0.05], "on top of")
        # Stacks gripped object on top of object at [0.25, 0.05]

        place_object([0.18, -0.10], None)
        # Places object at exact coordinates [0.18, -0.10]

        # Complete workflow:
        # pick_object("cube", [0.15, -0.05])
        # place_object([0.20, 0.10], "right next to")

    Args:
        place_coordinate (List[float]): Target coordinates [x, y] in meters where
            the object should be placed. Must be within workspace bounds.
        location (Union[Location, str, None]): Relative placement position. Options:
            - "left next to": Place to the left (increases Y)
            - "right next to": Place to the right (decreases Y)
            - "above": Place above (increases X, farther from robot)
            - "below": Place below (decreases X, closer to robot)
            - "on top of": Stack on top of object at coordinate
            - "inside": Place inside container at coordinate
            - "close to": Place within 2cm radius
            - None: Use exact coordinate (default)

    Returns:
        str: Success message with placement coordinates and location,
            or error description if placement failed.

    Note:
        Must call pick_object() before calling this function. For complete
        pick-and-place in one operation, use pick_place_object() instead.
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
    Push a specific object to a new position using the robot gripper.

    Use this function when an object is too large to grasp (width > 5cm). The robot
    approaches the object from the appropriate side and pushes it in the specified direction.

    Examples:
        push_object("large box", [0.25, 0.05], "right", 50.0)
        # Pushes large box 50mm (5cm) to the right from its current position

        push_object("book", [0.20, -0.03], "up", 30.0)
        # Pushes book 30mm upward (away from robot, increases X)

        push_object("tray", [0.18, 0.08], "left", 40.0)
        # Pushes tray 40mm to the left (increases Y)

        # Direction mapping:
        # "up"    → pushes away from robot (increases X coordinate)
        # "down"  → pushes toward robot (decreases X coordinate)
        # "left"  → pushes left (increases Y coordinate)
        # "right" → pushes right (decreases Y coordinate)

    Args:
        object_name (str): Name of the object to push. Ensure the name matches
            an object in the robot's environment. Case-sensitive.
        push_coordinate (List[float]): World coordinates [x, y] in meters where
            the object is currently located. Get from get_detected_objects().
        direction (str): Direction to push the object. Must be one of:
            "up", "down", "left", "right".
        distance (float): Distance to push in millimeters. Typical range: 10.0 to 100.0.
            Note: This is in millimeters, not meters (50.0 = 5cm).

    Returns:
        str: Success message with object name, starting position, direction, and distance,
            or error description if push operation failed.

    Note:
        Use push_object() instead of pick_place_object() when object width exceeds
        gripper capacity (~5cm). The robot approaches from the opposite side of the
        push direction to avoid collisions.
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
    Pick an object and move it a specified distance in a given direction.

    This is a convenience function that combines pick and place operations,
    calculating the target position based on direction and distance.

    Examples:
        move2by("pencil", [-0.11, 0.21], "left", 0.02)
        # Picks pencil at [-0.11, 0.21] and moves it 2cm left to [-0.11, 0.23]

        move2by("cube", [-0.11, 0.21], "up", 0.03)
        # Moves cube 3cm upward (increases X coordinate)

        move2by("pen", [0.15, -0.05], "right", 0.04, z_offset=0.02)
        # Picks pen with 2cm z-offset and moves it 4cm right

        # Direction mapping:
        # "left"  → increases Y (toward positive Y)
        # "right" → decreases Y (toward negative Y)
        # "up"    → increases X (away from robot)
        # "down"  → decreases X (toward robot)

    Args:
        object_name (str): Name of object to move. Must match detection exactly.
        pick_coordinate (List[float]): Current world coordinates [x, y] in meters
            where the object is located.
        direction (str): Direction to move. Must be one of: "left", "right", "up", "down".
        distance (float): Distance in meters to move the object.
            Typical range: 0.01 to 0.10 (1cm to 10cm).
        z_offset (float): Height offset in meters for picking (default: 0.001).
            Use higher values (e.g., 0.02) if object is on top of another object.

    Returns:
        str: Success message with pick and place coordinates, or error description.
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

    The gripper hovers over the workspace at optimal height and angle for
    object detection. Must be called before picking or placing objects.

    Examples:
        move2observation_pose("niryo_ws")
        # Moves to observation pose above niryo_ws workspace

        move2observation_pose("gazebo_1")
        # Moves to observation pose for simulation workspace

        # Typical workflow:
        # 1. move2observation_pose("niryo_ws")
        # 2. get_detected_objects()
        # 3. pick_place_object(...)
        # 4. move2observation_pose("niryo_ws")  # Return to home

    Args:
        workspace_id (str): ID of the workspace (e.g., "niryo_ws", "gazebo_1").
            Must match a configured workspace in the system.

    Returns:
        str: Success message confirming movement to observation pose,
            or error description if movement failed.

    Note:
        The robot automatically moves to observation pose before detection
        operations, but you may need to call this explicitly for other tasks.
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
    Reset the internal collision detection flag of the Niryo robot.

    Call this after a collision event to resume normal operation. This is
    Niryo-specific functionality.

    Examples:
        clear_collision_detected()
        # Clears collision flag
        # Returns: "✓ Collision detection flag cleared"

        # Recovery workflow:
        # 1. Collision occurs
        # 2. clear_collision_detected()
        # 3. calibrate()  # Optional
        # 4. Resume operations

        # After unexpected stop:
        # clear_collision_detected()
        # move2observation_pose("niryo_ws")

    Returns:
        str: Success message if flag cleared, or error description if failed.

    Note:
        This is a Niryo-specific function. May not work with other robot types.
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

    This procedure moves each joint to its home position and resets the robot's
    internal coordinate system. Should be called after power-up or if positioning
    errors are observed.

    Examples:
        calibrate()
        # Robot moves through calibration sequence
        # Returns: "✓ Robot calibration completed successfully"

        # Typical usage:
        # 1. Power on robot
        # 2. calibrate()
        # 3. Proceed with normal operations

        # After collision:
        # 1. clear_collision_detected()
        # 2. calibrate()
        # 3. Resume tasks

    Returns:
        str: Success message if calibration completed, or error description if failed.
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

    Supports optional spatial filtering (left/right/above/below) and label filtering.
    The robot automatically moves to observation pose before detection.

    Examples:
        get_detected_objects()
        # Returns: All detected objects in workspace

        get_detected_objects(location="close to", coordinate=[0.2, 0.0])
        # Returns: Objects within 2cm of [0.2, 0.0]

        get_detected_objects(label="pencil")
        # Returns: All objects labeled "pencil"

        get_detected_objects(location="left next to", coordinate=[0.20, 0.0], label="cube")
        # Returns: Cubes to the left of [0.20, 0.0]

    Args:
        location (Union[Location, str]): Spatial filter relative to coordinate. Options:
            - "left next to": Objects left of coordinate (positive Y)
            - "right next to": Objects right of coordinate (negative Y)
            - "above": Objects above coordinate (positive X)
            - "below": Objects below coordinate (negative X)
            - "close to": Objects within 2cm radius
            - None: No spatial filter (default)
        coordinate (Optional[List[float]]): Reference coordinate [x, y] in meters.
            Required if location is specified. Must be within workspace bounds.
        label (Optional[str]): Filter by object name. Case-sensitive, must match
            detection exactly (e.g., "pencil", not "Pencil").

    Returns:
        str: JSON string of detected objects with positions, dimensions, and orientations,
            or message if no objects match the criteria.
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
    Retrieve a detected object at or near a specified world coordinate.

    This method checks for objects detected by the camera that are close to the
    specified coordinate (within 2 centimeters). If multiple objects meet the
    criteria, the first object in the list is returned.

    Examples:
        get_detected_object([0.18, -0.05])
        # Returns: Object at [0.180, -0.050] (any object type)

        get_detected_object([0.18, -0.05], label="pen")
        # Returns: Only a "pen" object at that location

        get_detected_object([0.25, 0.0])
        # Returns: Object near workspace center

    Args:
        coordinate (List[float]): A 2D coordinate in the world coordinate system [x, y].
            Only objects within a 2-centimeter radius of this coordinate are considered.
        label (Optional[str]): An optional filter for the object's label. If specified,
            only an object with the matching label is returned. Case-sensitive.

    Returns:
        str: JSON string of the found object with position and dimensions,
            or message if no object found at that location.
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
    Return the largest detected object based on its area in square meters.

    The robot moves to observation pose, detects all objects, and identifies
    the one with the largest area (width × height).

    Examples:
        get_largest_detected_object()
        # Returns: {"label": "blue square", "position": {...}, ...}

        # Use result for manipulation:
        # 1. largest = get_largest_detected_object()
        # 2. Parse JSON to get position
        # 3. pick_object(largest["label"], ...)

    Returns:
        str: JSON string with largest object information including label, position,
            dimensions, and area. Returns "No objects detected" if workspace is empty.
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
    Return the smallest detected object based on its area in square meters.

    The robot moves to observation pose, detects all objects, and identifies
    the one with the smallest area (width × height).

    Examples:
        get_smallest_detected_object()
        # Returns: {"label": "pen", "position": {...}, ...}

        # Use for precise picking:
        # 1. smallest = get_smallest_detected_object()
        # 2. Extract coordinates
        # 3. pick_object(smallest["label"], ...)

    Returns:
        str: JSON string with smallest object information including label, position,
            dimensions, and area. Returns "No objects detected" if workspace is empty.
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
    Get detected objects sorted by size (area in square meters).

    The robot moves to observation pose and detects all objects, then sorts
    them by their area (width × height).

    Examples:
        get_detected_objects_sorted(ascending=True)
        # Returns: [smallest object, medium object, largest object]

        get_detected_objects_sorted(ascending=False)
        # Returns: [largest object, medium object, smallest object]

        get_detected_objects_sorted()
        # Returns: Objects sorted smallest to largest (default)

    Args:
        ascending (bool): If True, sort smallest to largest. If False, sort
            largest to smallest. Default is True.

    Returns:
        str: JSON string of sorted objects with positions and sizes,
            or message if no objects detected.
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
    """
    Make the robot speak a message using text-to-speech.

    The robot will announce the provided text through its audio output system,
    providing audible feedback to users.

    Examples:
        speak("I have picked up the pencil")
        # Robot says: "I have picked up the pencil"

        speak("Task completed successfully")
        # Robot announces task completion

        speak("Warning: Object detected in workspace")
        # Robot provides audio warning

    Args:
        text (str): The message to speak. Must be a non-empty string.
            Can include natural language sentences.

    Returns:
        str: Confirmation message with the text being spoken, or error if failed.
    """
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

    This function updates the task text that appears in video overlays,
    allowing viewers to see what command the user has given.

    Examples:
        set_user_task("Pick up the red cube")
        # Video overlay shows: "Task: Pick up the red cube"

        set_user_task("Organize workspace by color")
        # Video overlay updates to show new task

        set_user_task("Sort objects by size")
        # Task text changes in video feed

    Args:
        task (str): The user's natural language task or command.
            This will be displayed in video recordings.

    Returns:
        str: Confirmation message that the task has been set.
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
        if env is not None:
            env.cleanup()

        logger.info("=" * 80)
        logger.info("SERVER STOPPED")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

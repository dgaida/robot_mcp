# fastmcp_robot_server.py
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
    PickObjectInput,
    PickPlaceInput,
    PlaceObjectInput,
    PushObjectInput,
    WorkspacePointInput,
)

# ============================================================================
# VALIDATION DECORATOR
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


# ============================================================================
# LOGGING SETUP - Configure BEFORE any imports that might use logging
# ============================================================================

# Create log directory if it doesn't exist
os.makedirs("log", exist_ok=True)

# Single log file for entire system
log_filename = os.path.join("log", f'mcp_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure root logger - this ensures ALL loggers use the same file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        # DO NOT add StreamHandler - interferes with MCP communication
    ],
    force=True,  # Override any existing configuration
)

# Get logger for this module
logger = logging.getLogger("FastMCPRobotServer")

# Also set level for robot_environment package
logging.getLogger("RobotUniversalMCPClient").setLevel(logging.INFO)
logging.getLogger("robot_environment").setLevel(logging.INFO)
logging.getLogger("visual_detect_segment").setLevel(logging.INFO)

# Store log filename for reference
LOG_FILE = log_filename

logger.info("=" * 80)
logger.info(f"FastMCP Robot Server starting - Log file: {log_filename}")
logger.info("=" * 80)


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
# ENVIRONMENT TOOLS
# ============================================================================


@mcp.tool
@log_tool_call
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
@log_tool_call
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
@log_tool_call
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
@log_tool_call
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
@log_tool_call
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
    function and not robot_pick_object() followed by robot_place_object().

    Example call:

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
    --> Picks the cube with a 2cm z-offset (useful if it's on top of another object).

    Args:
        object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
        the robot's workspace.
        pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
        coordinates to identify the object's exact position.
        place_coordinate (List): The world coordinates [x, y] where the object should be placed at.
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
@log_tool_call
def move2by(
    object_name: str,
    pick_coordinate: List[float],
    direction: str,
    distance: float,
    z_offset: float = 0.001,
) -> str:
    """
    Command the pick-and-place robot arm to pick a specific object and move it a given distance left/right/up/down
    its original position using its gripper.
    The gripper will move to the specified 'pick_coordinate' and pick the named object.
    Then it will move the given distance in the direction specified by 'direction' and place the object there.

    Example call:

    robot.move2by(
        object_name='pencil',
        pick_coordinate=[-0.11, 0.21],
        direction='left',
        distance=0.02
    )
    --> Picks the pencil that is located at world coordinates [-0.11, 0.21] and places it 2 cm to the left
    at world coordinate [-0.11, 0.23].

    robot.move2by(
        object_name='cube',
        pick_coordinate=[-0.11, 0.21],
        direction='up',
        distance=0.03,
        z_offset=0.02
    )
    --> Picks the cube with a 2cm z-offset and places it 3cm upwards. The argument z_offset is useful if the cube is
    on top of, or blocking, another object. This is because the gripper then hovers 2cm above the workspace and does
    not touch the object beneath the cube. This assumes that the height of the object beneath does not exceed 2cm.
    Otherwise, select a larger z-offset.

    Args:
        object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
        the robot's workspace.
        pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
        coordinates to identify the object's exact position.
        direction (str): Direction to which object is moved. Possible values are: 'left', 'right', 'up', 'down'.
        distance (float): Specifies the relative distance in meters to which the object should be moved.
        z_offset (float): Additional height offset in meters to apply when picking (default: 0.001).
        Useful for picking objects that are stacked on top of other objects or are blocking other objects.

    Returns:
        str: Success message or error description
    """
    try:
        place_coordinate = pick_coordinate.copy()
        if direction == 'left':
            place_coordinate[1] += distance
        elif direction == 'right':
            place_coordinate[1] -= distance
        elif direction == 'up':
            place_coordinate[0] += distance
        elif direction == 'down':
            place_coordinate[0] -= distance
        else:
            raise ValueError(f"Invalid direction '{direction}'")

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
                f"and placed it {distance*100:.1f} centimeters {direction}wards at [{place_coordinate[0]:.3f}, {place_coordinate[1]:.3f}]"
            )
        else:
            return f"❌ Failed to pick and move '{object_name}'"
    except Exception as e:
        return f"❌ Error during move2by: {str(e)}"


@mcp.tool
@log_tool_call
@validate_input(PickObjectInput)
def pick_object(object_name: str, pick_coordinate: List[float], z_offset: float = 0.001) -> str:
    """
    Command the pick-and-place robot arm to pick up a specific object using its gripper. The gripper will move to
    the specified 'pick_coordinate' and pick the named object.

    Example call:

    robot.pick_object("pen", [0.01, -0.15])
    --> Picks the pen that is located at world coordinates [0.01, -0.15].

    robot.pick_object("pen", [0.01, -0.15], z_offset=0.02)
    --> Picks the pen with a 2cm offset above its detected position (useful for stacked objects).

    Args:
        object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
        the robot's workspace.
        pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
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


# TODO: Location not documented as Location
@mcp.tool
@log_tool_call
@validate_input(PlaceObjectInput)
def place_object(place_coordinate: List[float], location: Union[Location, str, None] = None) -> str:
    """
    Instruct the pick-and-place robot arm to place a picked object at the specified 'place_coordinate'. The
    function moves the gripper to the specified 'place_coordinate' and calculates the exact placement position from
    the given 'location'. Before calling this function you have to call robot_pick_object() to pick an object.

    Example call:

    robot.place_object([0.2, 0.0], "left next to")
    --> Places the already gripped object left next to the world coordinate [0.2, 0.0].

    Args:
        place_coordinate: The world coordinates [x, y] of the target object.
        location (str): Specifies the relative placement position of the picked object in relation to an object
        being at the 'place_coordinate'. Possible positions: 'left next to', 'right next to', 'above', 'below',
        'on top of', 'inside', or None. Set to None, if there is no location given in the task.
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
@log_tool_call
@validate_input(PushObjectInput)
def push_object(object_name: str, push_coordinate: List[float], direction: str, distance: float) -> str:
    """
    Direct the pick-and-place robot arm to push a specific object to a new position.
    This function should only be called if it is not possible to pick the object.
    An object cannot be picked if its shorter side is larger than the gripper.

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
@log_tool_call
def move2observation_pose(workspace_id: str) -> str:
    """
    The robot will move to a pose where it can observe (the gripper hovers over) the workspace given by workspace_id.
    Before a robot can pick up or place an object in a workspace, it must first move to this observation pose of the corresponding workspace.

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
@log_tool_call
def clear_collision_detected() -> str:
    """
    Reset the internal flag "collision_detected" of the Niryo robot. You need to call this after a
    collision of the robot.

    Returns:
        str: Success message
    """
    try:
        robot.robot().robot_ctrl().clear_collision_detected()
        return "✓ Collision detection flag cleared"
    except Exception as e:
        return f"❌ Error clearing collision flag: {str(e)}"


@mcp.tool
@log_tool_call
def calibrate() -> str:
    """
    Calibrate the robot.

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
# OBJECT DETECTION TOOLS
# ============================================================================


@mcp.tool
@log_tool_call
@validate_input(GetDetectedObjectsInput)
def get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: Optional[List[float]] = None,
    label: Optional[str] = None,
) -> str:
    """
    Get list of objects detected by the camera in the workspace.

    Args:
        location (Location, optional): acts as filter. can have the values:
        - "left next to": Only objects left of the given coordinate are returned,
        - "right next to": Only objects right of the given coordinate are returned,
        - "above": Only objects above the given coordinate are returned,
        - "below": Only objects below the given coordinate are returned,
        - "close to": Only objects close to the given coordinate are returned (within 2 centimeters),
        - None: no filter, all objects are returned (default).
        coordinate (List[float], optional): some (x,y) coordinate in the world coordinate system.
        Together with 'location' it acts as a filter. Required if 'location' is specified.
        label (str, optional): Only objects with the given label are returned.

    Returns:
        str: JSON string of detected objects or error message
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
@log_tool_call
def get_detected_object(coordinate: List[float], label: Optional[str] = None) -> str:
    """
    Retrieves a detected object at or near a specified world coordinate, optionally filtering by label.

    This method checks for objects detected by the camera that are close to the specified coordinate (within
    2 centimeters). If multiple objects meet the criteria, the first object in the list is returned.

    Args:
        coordinate (List[float]): A 2D coordinate in the world coordinate system [x, y].
            Only objects within a 2-centimeter radius of this coordinate are considered.
        label (Optional[str]): An optional filter for the object's label. If specified, only an object
            with the matching label is returned.

    Returns:
        str: The first object detected near the given coordinate (and matching the label, if provided). or error message
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
@log_tool_call
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
@log_tool_call
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
@log_tool_call
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


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


# python server/fastmcp_robot_server.py
# für realen Roboter
# python server/fastmcp_robot_server.py --no-simulation
def main():
    """Main entry point when running as script."""
    parser = argparse.ArgumentParser(description="FastMCP Robot Server with Pydantic Validation")
    parser.add_argument("--robot", choices=["niryo", "widowx"], default="niryo")
    parser.add_argument("--no-simulation", action="store_false", dest="simulation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--verbose", action="store_true")

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
    logger.info(f"  Log File:     {log_filename}")
    logger.info("=" * 80)

    # Print to console (since logging is file-only)
    print("=" * 60)
    print("STARTING FASTMCP ROBOT SERVER (with Pydantic Validation)")
    print("=" * 60)
    print(f"Robot:        {args.robot}")
    print(f"Simulation:   {args.simulation}")
    print(f"Host:         {args.host}")
    print(f"Port:         {args.port}")
    print(f"Camera:       {not args.no_camera}")
    print(f"Log File:     {log_filename}")
    print("=" * 60)
    print(f"\nServer running at: http://{args.host}:{args.port}")
    print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    print(f"\nMonitor logs: tail -f {log_filename}")
    print("\n✨ Pydantic validation enabled for all tool inputs")
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

    logger.info("Starting MCP server with Pydantic validation...")

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

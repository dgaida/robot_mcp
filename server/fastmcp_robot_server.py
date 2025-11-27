# fastmcp_robot_server.py
import argparse
import functools
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

from fastmcp import FastMCP
from robot_environment import Environment
from robot_workspace import Location

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
        logging.FileHandler(log_filename),
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

# ============================================================================
# MCP SETUP
# ============================================================================

# Init MCP
mcp = FastMCP("robot-environment")

# Global environment - initialized later
env = None
robot = None


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
def get_largest_free_space_with_center() -> tuple[float, float, float]:
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
        tuple: (largest_free_area_m2, center_x, center_y) where:
            - largest_free_area_m2 (float): Largest free area in square meters.
            - center_x (float): X-coordinate of the center of the largest free area in meters.
            - center_y (float): Y-coordinate of the center of the largest free area in meters.
    """
    return env.get_largest_free_space_with_center()


@mcp.tool
@log_tool_call
def get_workspace_coordinate_from_point(workspace_id: str, point: str) -> Optional[List[float]]:
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
        List[float]: (x,y) world coordinate of the point on the workspace that was specified by the argument point.
    """
    return env.get_workspace_coordinate_from_point(workspace_id, point)


@mcp.tool
@log_tool_call
def get_object_labels_as_string() -> str:
    """
    Returns all object labels that the object detection model is able to detect as a comma separated string.
    Call this method if the user wants to know which objects the robot can pick or is able to detect.

    Returns:
        str: "chocolate bar, blue box, cigarette, ..."
    """
    return env.get_object_labels_as_string()


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
    return env.add_object_name2object_labels(object_name)


# ============================================================================
# ROBOT TOOLS
# ============================================================================


@mcp.tool
@log_tool_call
def pick_place_object(
    object_name: str,
    pick_coordinate: list[float],
    place_coordinate: list[float],
    location: Union[Location, str, None] = None,
) -> bool:
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

    Args:
        object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
        the robot's workspace.
        pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
        coordinates to identify the object's exact position.
        place_coordinate (List): The world coordinates [x, y] where the object should be placed at.
        location (Location): Specifies the relative placement position of the picked object in relation to an object
        being at the 'place_coordinate'. Possible values are defined in the `Location` Enum:
            - `Location.LEFT_NEXT_TO`: Left of the reference object.
            - `Location.RIGHT_NEXT_TO`: Right of the reference object.
            - `Location.ABOVE`: Above the reference object.
            - `Location.BELOW`: Below the reference object.
            - `Location.ON_TOP_OF`: On top of the reference object.
            - `Location.INSIDE`: Inside the reference object.
            - `Location.NONE`: No specific location relative to another object.

    Returns:
        bool: Always returns `True` after the pick-and-place operation.
    """
    return robot.pick_place_object(
        object_name=object_name,
        pick_coordinate=pick_coordinate,
        place_coordinate=place_coordinate,
        location=location,
    )


@mcp.tool
@log_tool_call
def pick_object(object_name: str, pick_coordinate: List) -> bool:
    """
    Command the pick-and-place robot arm to pick up a specific object using its gripper. The gripper will move to
    the specified 'pick_coordinate' and pick the named object.

    Example call:

    robot.pick_object("pen", [0.01, -0.15])
    --> Picks the pen that is located at world coordinates [0.01, -0.15].

    Args:
        object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
        the robot's workspace.
        pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
        coordinates to identify the object's exact position.
    Returns:
        bool: True on success
    """
    return robot.pick_object(object_name=object_name, pick_coordinate=pick_coordinate)


@mcp.tool
@log_tool_call
def place_object(place_coordinate: List, location: Union[Location, str, None] = None) -> bool:
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
        bool: True on success
    """
    return robot.place_object(place_coordinate=place_coordinate, location=location)


@mcp.tool
@log_tool_call
def push_object(object_name: str, push_coordinate: List, direction: str, distance: float):
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
        bool: True on success
    """
    return robot.push_object(object_name, push_coordinate, direction, distance)


@mcp.tool
@log_tool_call
def move2observation_pose(workspace_id: str) -> None:
    """
    The robot will move to a pose where it can observe (the gripper hovers over) the workspace given by workspace_id.
    Before a robot can pick up or place an object in a workspace, it must first move to this observation pose of the corresponding workspace.

    Args:
        workspace_id: ID of the workspace

    Returns:
        None
    """
    return robot.move2observation_pose(workspace_id)


@mcp.tool
@log_tool_call
def clear_collision_detected() -> None:
    """
    Reset the internal flag "collision_detected" of the Niryo robot. You need to call this after a
    collision of the robot.

    Returns:
        None
    """
    robot.robot().robot_ctrl().clear_collision_detected()


# ============================================================================
# OBJECT DETECTION TOOLS
# ============================================================================


@mcp.tool
@log_tool_call
def get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: List[float] = None,
    label: Optional[str] = None,
) -> Optional[List[Dict]]:
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
        Optional[List[Dict]]: list of objects detected by the camera in the workspace.
    """
    env.robot_move2home_observation_pose()

    # wait for robot to reach observation pose
    time.sleep(1)

    detected_objects = env.get_detected_objects()
    objects = detected_objects.get_detected_objects_serializable(location, coordinate, label)
    return objects


@mcp.tool
@log_tool_call
def get_detected_object(coordinate: List[float], label: Optional[str] = None) -> Optional[Dict]:
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
        Optional[Dict]: The first object detected near the given coordinate (and matching the label, if provided).
        Returns `None` if no such object is found.
    """
    env.robot_move2home_observation_pose()

    detected_objects = env.get_detected_objects()
    return detected_objects.get_detected_object(coordinate, label, True)


@mcp.tool
@log_tool_call
def get_largest_detected_object() -> tuple[List[Dict], float]:
    """
    Returns the largest detected object based on its size in square meters.

    Returns:
        tuple: (largest_object, largest_size_m2) where:
            - largest_object (List[Dict]): The largest detected object.
            - largest_size_m2 (float): The size of the largest object in square meters.
    """
    env.robot_move2home_observation_pose()

    detected_objects = env.get_detected_objects()
    return detected_objects.get_largest_detected_object(True)


@mcp.tool
@log_tool_call
def get_smallest_detected_object() -> tuple[List[Dict], float]:
    """
    Returns the smallest detected object based on its size in square meters.

    Returns:
        tuple: (smallest_object, smallest_size_m2) where:
            - smallest_object (List[Dict]): The smallest detected object.
            - smallest_size_m2 (float): The size of the smallest object in square meters.
    """
    env.robot_move2home_observation_pose()

    detected_objects = env.get_detected_objects()
    return detected_objects.get_smallest_detected_object(True)


@mcp.tool
@log_tool_call
def get_detected_objects_sorted(ascending: bool = True) -> List[Dict]:
    """
    Returns the detected objects sorted by size in square meters.

    Args:
        ascending (bool): If True, sorts the objects in ascending order.
                          If False, sorts in descending order.

    Returns:
        List[Dict]: The list of detected objects sorted by size.
    """
    env.robot_move2home_observation_pose()

    detected_objects = env.get_detected_objects()
    return detected_objects.get_detected_objects_sorted(ascending, True)


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
#     return myobject.label()


# ============================================================================
# FEEDBACK TOOLS
# ============================================================================


@mcp.tool
@log_tool_call
def speak(text: str) -> str:
    """Make the robot speak a message using text-to-speech."""
    env.oralcom_call_text2speech_async(text)
    return f"Speaking: {text}"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


# python server/fastmcp_robot_server.py
# für realen Roboter
# python server/fastmcp_robot_server.py --no-simulation
def main():
    """Main entry point when running as script."""
    parser = argparse.ArgumentParser(description="FastMCP Robot Server")
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
    print("STARTING FASTMCP ROBOT SERVER")
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

    logger.info("Starting MCP server...")

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

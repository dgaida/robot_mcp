# fastmcp_robot_server.py
from typing import TYPE_CHECKING, Optional, List, Union

from fastmcp import FastMCP
from robot_environment import Environment
from robot_environment.robot.robot_api import Location
from robot_environment.objects.objects import Objects

# if TYPE_CHECKING:
#     from robot_environment.robot.robot_api import Location

import logging
import os
from datetime import datetime


# Configure logging to file (NOT to stdout/stderr!)
log_filename = os.path.join('log', f'mcp_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        # DO NOT add StreamHandler - it would interfere with MCP communication
    ]
)
logger = logging.getLogger("FastMCPRobotServer")

# Init MCP
mcp = FastMCP("robot-environment")

# Environment-Setup (vereinfacht, Parameter kannst du anpassen)
env = Environment(el_api_key="", use_simulation=True, robot_id="niryo", verbose=False, start_camera_thread=True)
robot = env.robot()


@mcp.tool
def pick_place_object(object_name: str, pick_coordinate: list[float], place_coordinate: list[float],
                      location: Union[Location, str, None] = None) -> bool:
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
    return robot.pick_place_object(object_name=object_name, pick_coordinate=pick_coordinate,
                                   place_coordinate=place_coordinate, location=location)


@mcp.tool
def move2observation_pose(workspace_id: str) -> None:
    """
    The robot will move to a pose where it can observe (the gripper hovers over) the workspace given by workspace_id.
    Before a robot can pick up or place an object in a workspace, it must first move to this observation pose of the corresponding workspace.

    Args:
        workspace_id: id of the workspace

    Returns:
        None
    """
    return robot.move2observation_pose(workspace_id)


@mcp.tool
def get_detected_objects(location: Union[Location, str] = Location.NONE, coordinate: List[float] = None,
                         label: Optional[str] = None) -> Optional[Objects]:
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
        Optional["Objects"]: list of objects detected by the camera in the workspace.
    """
    detected_objects = env.get_detected_objects()

    detected_obj = detected_objects.get_detected_objects(location, coordinate, label)

    objects = Objects.objects_to_dict_list(detected_obj)

    # print("***********")
    # print(objects)
    # print("***********")

    return objects

@mcp.tool
def speak(text: str) -> str:
    """Make the robot speak a message using text-to-speech."""
    env.oralcom_call_text2speech_async(text)
    return f"Speaking: {text}"


# Weitere Tools (push_object, get_detected_objects, etc.) kannst du analog hinzufügen

# python server/fastmcp_robot_server.py
if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000)

"""
Model Context Protocol Server for Robot Environment Control

This MCP server exposes robot control and object detection capabilities
through the Model Context Protocol, enabling natural language control of
robotic pick-and-place operations.

Install dependencies:
    pip install mcp

Usage:
    python mcp_robot_server.py
"""

import asyncio
import json
from typing import Any, Sequence
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import robot environment components
from robot_environment import Environment  # , Objects
from robot_environment.robot.robot_api import Location

import logging
import sys
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
logger = logging.getLogger("RobotMCPServer")


class RobotMCPServer:
    """MCP Server wrapper for robot environment control."""
    
    def __init__(self, el_api_key: str, use_simulation: bool = False, 
                 robot_id: str = "niryo", verbose: bool = True):
        """
        Initialize the robot MCP server.
        
        Args:
            el_api_key: ElevenLabs API key for text-to-speech
            use_simulation: Whether to use simulation mode
            robot_id: Robot identifier ("niryo" or "widowx")
            verbose: Enable verbose logging
        """
        logger.info("=" * 50)
        logger.info("Initializing MCP Robot Server")
        logger.info(f"Robot: {robot_id}")
        logger.info(f"Simulation: {use_simulation}")
        logger.info(f"Verbose: {verbose}")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 50)

        self.env = Environment(el_api_key, use_simulation, robot_id, verbose=False, start_camera_thread=False)
        self.robot = self.env.robot()
        self.server = Server("robot-environment")

        self.available_tools = self._create_tool_definitions()

        # Request Counter für Debugging
        self.request_counter = 0

        self._setup_handlers()

        logger.info("MCP Server initialized successfully!")
        logger.info(f"Available tools: {len(self.available_tools)}")

    def _create_tool_definitions(self) -> list[Tool]:
        """Create tool definitions (separated for clarity)."""
        logger.debug("Creating tool definitions...")

        tools = [
            Tool(
                name="pick_place_object",
                description=(
                    "Pick up a specific object and place it at a target location. "
                    "This is the primary function for moving objects. "
                    "The robot will move to the pick coordinate, grasp the object, "
                    "move to the place coordinate, and release it."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name/label of the object to pick (e.g., 'pencil', 'red cube')"
                        },
                        "pick_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "World coordinates [x, y] in meters where the object is located"
                        },
                        "place_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "World coordinates [x, y] in meters where to place the object"
                        },
                        "location": {
                            "type": "string",
                            "enum": [
                                "left next to",
                                "right next to",
                                "above",
                                "below",
                                "on top of",
                                "inside",
                                "close to",
                                "none"
                            ],
                            "description": "Relative placement position to another object at place_coordinate",
                            "default": "none"
                        }
                    },
                    "required": ["object_name", "pick_coordinate", "place_coordinate"]
                }
            ),
            Tool(
                name="pick_object",
                description=(
                    "Pick up a specific object using the robot gripper. "
                    "Use this when you only want to pick without immediately placing. "
                    "Must be followed by place_object to complete the operation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name/label of the object to pick"
                        },
                        "pick_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "World coordinates [x, y] in meters"
                        }
                    },
                    "required": ["object_name", "pick_coordinate"]
                }
            ),
            Tool(
                name="place_object",
                description=(
                    "Place a previously picked object at a target location. "
                    "Must be called after pick_object. "
                    "The location parameter specifies placement relative to an object at place_coordinate."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "place_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "World coordinates [x, y] in meters"
                        },
                        "location": {
                            "type": "string",
                            "enum": [
                                "left next to",
                                "right next to",
                                "above",
                                "below",
                                "on top of",
                                "inside",
                                "none"
                            ],
                            "description": "Relative placement position",
                            "default": "none"
                        }
                    },
                    "required": ["place_coordinate"]
                }
            ),
            Tool(
                name="push_object",
                description=(
                    "Push an object in a specified direction. "
                    "Use this only when the object cannot be picked (e.g., too large for gripper). "
                    "The object will be pushed from its current position."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name/label of the object to push"
                        },
                        "push_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "World coordinates [x, y] in meters where the object is located"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to push the object"
                        },
                        "distance": {
                            "type": "number",
                            "description": "Distance to push in millimeters",
                            "minimum": 0
                        }
                    },
                    "required": ["object_name", "push_coordinate", "direction", "distance"]
                }
            ),
            Tool(
                name="move_to_observation_pose",
                description=(
                    "Move the robot to an observation pose above a workspace. "
                    "This must be called before picking or placing objects in a workspace. "
                    "The robot will position itself to observe the entire workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the workspace (e.g., 'niryo_ws', 'gazebo_1')",
                            "default": "niryo_ws"
                        }
                    },
                    "required": ["workspace_id"]
                }
            ),
            Tool(
                name="get_detected_objects",
                description=(
                    "Get a list of all objects currently detected by the robot's camera. "
                    "Returns object names, positions, sizes, and orientations. "
                    "Use this to understand what objects are available in the workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label_filter": {
                            "type": "string",
                            "description": "Optional: filter by object label/name",
                            "default": None
                        },
                        "location_filter": {
                            "type": "string",
                            "enum": [
                                "left next to",
                                "right next to",
                                "above",
                                "below",
                                "close to",
                                "none"
                            ],
                            "description": "Optional: filter by location relative to a coordinate",
                            "default": "none"
                        },
                        "reference_coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Reference coordinate [x, y] for location filter",
                            "default": None
                        }
                    }
                }
            ),
            Tool(
                name="get_object_at_location",
                description=(
                    "Find a specific object at or near a given coordinate. "
                    "Returns detailed information about the object including "
                    "its exact position, size, and orientation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "World coordinates [x, y] in meters"
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional: object label to match",
                            "default": None
                        }
                    },
                    "required": ["coordinate"]
                }
            ),
            Tool(
                name="get_nearest_object",
                description=(
                    "Find the object nearest to a given coordinate. "
                    "Optionally filter by object label. "
                    "Returns the object and its distance from the coordinate."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "World coordinates [x, y] in meters"
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional: filter by object label",
                            "default": None
                        }
                    },
                    "required": ["coordinate"]
                }
            ),
            Tool(
                name="get_largest_object",
                description=(
                    "Find the largest object in the workspace by area. "
                    "Returns the object and its size in square centimeters."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_smallest_object",
                description=(
                    "Find the smallest object in the workspace by area. "
                    "Returns the object and its size in square centimeters."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_workspace_info",
                description=(
                    "Get information about a workspace including its dimensions, "
                    "center position, and observation pose. "
                    "Useful for understanding the working area."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the workspace",
                            "default": None
                        },
                        "workspace_index": {
                            "type": "integer",
                            "description": "Index of the workspace (0-based)",
                            "default": 0
                        }
                    }
                }
            ),
            Tool(
                name="speak",
                description=(
                    "Make the robot speak a message using text-to-speech. "
                    "Useful for providing audio feedback about robot actions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text message to speak"
                        }
                    },
                    "required": ["text"]
                }
            )
        ]

        logger.debug(f"Created {len(tools)} tool definitions")

        return tools

    def _setup_handlers(self):
        """Set up MCP server request handlers."""

        # Add request interceptor
        # @self.server.request()
        # async def handle_request(request):
        #     """Log all incoming requests."""
        #     self.request_counter += 1
        #     logger.info("=" * 60)
        #     logger.info(f"REQUEST #{self.request_counter} RECEIVED")
        #     logger.info(f"Method: {request.method if hasattr(request, 'method') else 'unknown'}")
        #     logger.info(f"Request type: {type(request).__name__}")
        #     logger.info(f"Request: {request}")
        #     logger.info("=" * 60)

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available robot control tools."""
            logger.info("list_tools() called by client")
            logger.debug(f"Returning {len(self.available_tools)} tools")
            return self.available_tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
            """Handle tool execution requests."""
            logger.info(f"call_tool() invoked: {name}")
            logger.debug(f"Arguments: {json.dumps(arguments, indent=2)}")

            try:
                if name == "pick_place_object":
                    result = self.robot.pick_place_object(
                        object_name=arguments["object_name"],
                        pick_coordinate=arguments["pick_coordinate"],
                        place_coordinate=arguments["place_coordinate"],
                        location=arguments.get("location", "none")
                    )
                    return [TextContent(
                        type="text",
                        text=f"Successfully picked '{arguments['object_name']}' from {arguments['pick_coordinate']} "
                             f"and placed it at {arguments['place_coordinate']} ({arguments.get('location', 'none')})"
                    )]
                
                elif name == "pick_object":
                    result = self.robot.pick_object(
                        object_name=arguments["object_name"],
                        pick_coordinate=arguments["pick_coordinate"]
                    )
                    return [TextContent(
                        type="text",
                        text=f"Successfully picked '{arguments['object_name']}' from {arguments['pick_coordinate']}"
                    )]
                
                elif name == "place_object":
                    result = self.robot.place_object(
                        place_coordinate=arguments["place_coordinate"],
                        location=arguments.get("location", "none")
                    )
                    return [TextContent(
                        type="text",
                        text=f"Successfully placed object at {arguments['place_coordinate']} "
                             f"({arguments.get('location', 'none')})"
                    )]
                
                elif name == "push_object":
                    result = self.robot.push_object(
                        object_name=arguments["object_name"],
                        push_coordinate=arguments["push_coordinate"],
                        direction=arguments["direction"],
                        distance=arguments["distance"]
                    )
                    return [TextContent(
                        type="text",
                        text=f"Successfully pushed '{arguments['object_name']}' {arguments['direction']} "
                             f"by {arguments['distance']}mm from {arguments['push_coordinate']}"
                    )]
                
                elif name == "move_to_observation_pose":
                    workspace_id = arguments.get("workspace_id", self.env.get_workspace_home_id())
                    self.env.robot_move2observation_pose(workspace_id)
                    return [TextContent(
                        type="text",
                        text=f"Robot moved to observation pose for workspace '{workspace_id}'"
                    )]
                
                elif name == "get_detected_objects":
                    detected_objects = self.env.get_detected_objects()
                    
                    # Apply filters if provided
                    label_filter = arguments.get("label_filter")
                    location_filter = arguments.get("location_filter", "none")
                    ref_coord = arguments.get("reference_coordinate")
                    
                    if label_filter or location_filter != "none":
                        detected_objects = detected_objects.get_detected_objects(
                            location=location_filter if location_filter != "none" else Location.NONE,
                            coordinate=ref_coord,
                            label=label_filter
                        )
                    
                    if len(detected_objects) == 0:
                        return [TextContent(type="text", text="No objects detected in the workspace.")]
                    
                    # Format object information
                    object_info = []
                    for obj in detected_objects:
                        info = {
                            "label": obj.label(),
                            "position": {"x": round(obj.x_com(), 3), "y": round(obj.y_com(), 3)},
                            "size": {
                                "width_m": round(obj.width_m(), 3),
                                "height_m": round(obj.height_m(), 3),
                                "area_cm2": round(obj.size_m2() * 10000, 2)
                            },
                            "orientation_rad": round(obj.gripper_rotation(), 3)
                        }
                        object_info.append(info)
                    
                    return [TextContent(
                        type="text",
                        text=f"Detected {len(object_info)} object(s):\n\n" + 
                             json.dumps(object_info, indent=2)
                    )]
                
                elif name == "get_object_at_location":
                    coordinate = arguments["coordinate"]
                    label = arguments.get("label")
                    
                    detected_objects = self.env.get_detected_objects()
                    obj = detected_objects.get_detected_object(coordinate, label)
                    
                    if obj is None:
                        return [TextContent(
                            type="text",
                            text=f"No object found at coordinate {coordinate}" + 
                                 (f" with label '{label}'" if label else "")
                        )]
                    
                    info = {
                        "label": obj.label(),
                        "position": {"x": round(obj.x_com(), 3), "y": round(obj.y_com(), 3)},
                        "size": {
                            "width_m": round(obj.width_m(), 3),
                            "height_m": round(obj.height_m(), 3),
                            "area_cm2": round(obj.size_m2() * 10000, 2)
                        },
                        "orientation_rad": round(obj.gripper_rotation(), 3)
                    }
                    
                    return [TextContent(
                        type="text",
                        text=f"Object found:\n\n{json.dumps(info, indent=2)}"
                    )]
                
                elif name == "get_nearest_object":
                    coordinate = arguments["coordinate"]
                    label = arguments.get("label")
                    
                    detected_objects = self.env.get_detected_objects()
                    obj, distance = detected_objects.get_nearest_detected_object(coordinate, label)
                    
                    if obj is None:
                        return [TextContent(
                            type="text",
                            text=f"No objects found" + (f" with label '{label}'" if label else "")
                        )]
                    
                    info = {
                        "label": obj.label(),
                        "position": {"x": round(obj.x_com(), 3), "y": round(obj.y_com(), 3)},
                        "distance_m": round(distance, 3),
                        "size": {
                            "width_m": round(obj.width_m(), 3),
                            "height_m": round(obj.height_m(), 3),
                            "area_cm2": round(obj.size_m2() * 10000, 2)
                        }
                    }
                    
                    return [TextContent(
                        type="text",
                        text=f"Nearest object:\n\n{json.dumps(info, indent=2)}"
                    )]
                
                elif name == "get_largest_object":
                    detected_objects = self.env.get_detected_objects()
                    
                    if len(detected_objects) == 0:
                        return [TextContent(type="text", text="No objects detected.")]
                    
                    obj, size = detected_objects.get_largest_detected_object()
                    
                    info = {
                        "label": obj.label(),
                        "position": {"x": round(obj.x_com(), 3), "y": round(obj.y_com(), 3)},
                        "size": {
                            "width_m": round(obj.width_m(), 3),
                            "height_m": round(obj.height_m(), 3),
                            "area_cm2": round(size * 10000, 2)
                        }
                    }
                    
                    return [TextContent(
                        type="text",
                        text=f"Largest object:\n\n{json.dumps(info, indent=2)}"
                    )]
                
                elif name == "get_smallest_object":
                    detected_objects = self.env.get_detected_objects()
                    
                    if len(detected_objects) == 0:
                        return [TextContent(type="text", text="No objects detected.")]
                    
                    obj, size = detected_objects.get_smallest_detected_object()
                    
                    info = {
                        "label": obj.label(),
                        "position": {"x": round(obj.x_com(), 3), "y": round(obj.y_com(), 3)},
                        "size": {
                            "width_m": round(obj.width_m(), 3),
                            "height_m": round(obj.height_m(), 3),
                            "area_cm2": round(size * 10000, 2)
                        }
                    }
                    
                    return [TextContent(
                        type="text",
                        text=f"Smallest object:\n\n{json.dumps(info, indent=2)}"
                    )]
                
                elif name == "get_workspace_info":
                    workspace_id = arguments.get("workspace_id")
                    workspace_index = arguments.get("workspace_index", 0)
                    
                    if workspace_id:
                        workspace = self.env.get_workspace_by_id(workspace_id)
                    else:
                        workspace = self.env.get_workspace(workspace_index)
                    
                    if workspace is None:
                        return [TextContent(type="text", text="Workspace not found.")]
                    
                    info = {
                        "id": workspace.id(),
                        "dimensions": {
                            "width_m": round(workspace.width_m(), 3),
                            "height_m": round(workspace.height_m(), 3)
                        },
                        "center_position": {
                            "x": round(workspace.xy_center_wc().x, 3),
                            "y": round(workspace.xy_center_wc().y, 3),
                            "z": round(workspace.xy_center_wc().z, 3)
                        },
                        "observation_pose": {
                            "x": round(workspace.observation_pose().x, 3),
                            "y": round(workspace.observation_pose().y, 3),
                            "z": round(workspace.observation_pose().z, 3),
                            "roll": round(workspace.observation_pose().roll, 3),
                            "pitch": round(workspace.observation_pose().pitch, 3),
                            "yaw": round(workspace.observation_pose().yaw, 3)
                        }
                    }
                    
                    return [TextContent(
                        type="text",
                        text=f"Workspace information:\n\n{json.dumps(info, indent=2)}"
                    )]
                
                elif name == "speak":
                    text = arguments["text"]
                    thread = self.env.oralcom_call_text2speech_async(text)
                    # Don't wait for speech to complete
                    return [TextContent(
                        type="text",
                        text=f"Speaking: '{text}'"
                    )]
                
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]

        logger.debug("MCP handlers registered successfully")

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting MCP server stdio communication...")

        try:
            # WICHTIG: stdio_server() erstellt die Streams für uns
            async with stdio_server() as (read_stream, write_stream):
                logger.info("stdio_server context entered successfully")
                logger.info(f"Read stream: {read_stream}")
                logger.info(f"Write stream: {write_stream}")

                # Server mit den Streams starten
                logger.info("Starting server.run()...")
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
                logger.info("server.run() completed")

        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise


async def main():
    """Main entry point for the MCP server."""
    import sys
    
    # Configuration - you can modify these or read from environment variables
    config = {
        "el_api_key": "",  # Your ElevenLabs API key (empty string to use Kokoro)
        "use_simulation": True,  # Set to True for Gazebo simulation
        "robot_id": "niryo",  # "niryo" or "widowx"
        "verbose": True
    }
    
    # Override from command line arguments if provided
    if len(sys.argv) > 1:
        config["robot_id"] = sys.argv[1]
    if len(sys.argv) > 2:
        config["use_simulation"] = sys.argv[2].lower() == "true"

    logger.info("=" * 60)
    logger.info("MCP ROBOT SERVER - STARTING")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info("=" * 60)

    server = RobotMCPServer(**config)

    logger.info("Calling server.run()...")
    await server.run()
    logger.info("Server shutdown complete")


# python server/mcp_robot_server.py niryo true
if __name__ == "__main__":
    try:
        # Write startup message to stderr (won't interfere with MCP stdio)
        sys.stderr.write("\n")
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write("MCP ROBOT SERVER\n")
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write(f"Log file: {log_filename}\n")
        sys.stderr.write("Server starting... (check log for details)\n")
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write("\n")
        sys.stderr.flush()

        # Run the server
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("\nServer stopped by user (Ctrl+C)")
        sys.stderr.write("\n✓ Server stopped\n")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.stderr.write(f"\n✗ Fatal error: {e}\n")
        sys.stderr.write(f"Check log file: {log_filename}\n")
        sys.exit(1)

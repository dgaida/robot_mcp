# Robot MCP - Detailed Documentation

This document provides in-depth information about the Robot MCP system architecture, implementation details, and advanced usage.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [FastMCP Implementation](#fastmcp-implementation)
- [Coordinate Systems](#coordinate-systems)
- [Object Detection](#object-detection)
- [Advanced Usage](#advanced-usage)
- [Performance Tuning](#performance-tuning)
- [Extending the System](#extending-the-system)

## Architecture Overview

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Client Layer                       │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ Groq LLM API │  │ FastMCP    │  │ Gradio GUI  │  │
│  │              │  │ Client     │  │             │  │
│  └──────────────┘  └────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                       │ SSE/HTTP
┌─────────────────────────────────────────────────────┐
│                  Server Layer                       │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ FastMCP      │  │ Tool       │  │ Robot       │  │
│  │ Server       │  │ Handlers   │  │ Environment │  │
│  └──────────────┘  └────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                       │ Python API
┌─────────────────────────────────────────────────────┐
│                 Hardware Layer                      │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ Robot Arm    │  │ Camera     │  │ Redis       │  │
│  │ (Niryo/WX)   │  │ (Vision)   │  │ (Comm)      │  │
│  └──────────────┘  └────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Repositories Architecture

![Repositories Architecture](robot_repos_architecture.png)

**References**

- [robot_environment](https://github.com/dgaida/robot_environment)
- [robot_workspace](https://github.com/dgaida/robot_workspace)
- [redis_robot_comm](https://github.com/dgaida/redis_robot_comm)
- [vision_detect_segment](https://github.com/dgaida/vision_detect_segment)
- [speech2text](https://github.com/dgaida/speech2text)
- [text2speech](https://github.com/dgaida/text2speech)

### Why FastMCP?

FastMCP offers several advantages over the original MCP protocol:

1. **HTTP/SSE Transport**: More flexible than stdio, easier to debug
2. **Modern Python**: Uses async/await patterns
3. **Better Error Handling**: Clearer error messages and stack traces
4. **Easier Development**: Simpler tool registration with decorators
5. **Network Ready**: Can run client and server on different machines

## System Components

### 1. FastMCP Client (`client/fastmcp_groq_client.py`)

**Purpose**: Bridge between Groq LLM and MCP server

**Key Features**:
- Maintains conversation history
- Converts tools to Groq function calling format
- Handles tool execution and result processing
- Supports interactive and programmatic modes

**Main Classes**:
- `RobotFastMCPClient` - Main client class

**Connection Flow**:
```python
client = RobotFastMCPClient(groq_api_key="key")
await client.connect()  # Connects to http://127.0.0.1:8000/sse
# Use client
await client.disconnect()
```

### 2. FastMCP Server (`server/fastmcp_robot_server.py`)

**Purpose**: Expose robot control as MCP tools

**Key Features**:
- Decorator-based tool registration
- Automatic schema generation
- Type-safe tool parameters
- Environment initialization

**Tool Registration**:
```python
from fastmcp import FastMCP
mcp = FastMCP("robot-environment")

@mcp.tool
def pick_place_object(
    object_name: str,
    pick_coordinate: list[float],
    place_coordinate: list[float],
    location: Union[Location, str, None] = None
) -> bool:
    """Pick and place an object."""
    return robot.pick_place_object(
        object_name, pick_coordinate,
        place_coordinate, location
    )
```

### 3. Robot Environment (`robot_environment` package)

**Purpose**: Hardware abstraction and vision processing

**Core Classes**:
- `Environment` - Main orchestrator
- `Robot` - High-level robot API
- `RobotController` - Hardware-specific control
- `VisualCortex` - Object detection
- `Objects` - Object collection and queries

**Initialization**:
```python
env = Environment(
    el_api_key="",           # For TTS
    use_simulation=False,     # Real robot
    robot_id="niryo",        # Robot type
    verbose=True,
    start_camera_thread=True # Background updates
)
```

### 4. Gradio Web Interface (`robot_gui/mcp_app.py`)

**Purpose**: User-friendly web interface

**Features**:
- Chat with robot assistant
- Live camera feed with annotations
- Voice input via Whisper
- Example task templates
- System status monitoring

**Architecture**:
```python
RobotMCPGUI
  ├─ RobotFastMCPClient (LLM integration)
  ├─ Speech2Text (voice input)
  ├─ RedisImageStreamer (camera feed)
  └─ Gradio Interface (UI)
```

## Data Flow

### Complete Pick-and-Place Workflow

```
1. USER INPUT
   User: "Pick up the pencil and place it next to the red cube"

2. LLM PROCESSING
   Groq LLM:
   ├─ Parses natural language
   ├─ Decides: Need to detect objects first
   └─ Generates tool calls

3. TOOL EXECUTION
   MCP Client → MCP Server:

   Call 1: get_detected_objects()
   ├─ Environment.get_detected_objects()
   ├─ Returns: [
   │    {label: "pencil", position: [0.15, -0.05]},
   │    {label: "red cube", position: [0.20, 0.10]}
   │  ]
   └─ Result sent back to LLM

   Call 2: pick_place_object(
      object_name="pencil",
      pick_coordinate=[0.15, -0.05],
      place_coordinate=[0.20, 0.10],
      location="right next to"
   )
   ├─ Robot.pick_place_object()
   ├─ RobotController executes motion
   ├─ Physical robot moves
   └─ Returns: True (success)

4. LLM RESPONSE
   Groq LLM:
   ├─ Synthesizes results
   └─ Generates: "Done! I've placed the pencil to the right
                  of the red cube."

5. USER FEEDBACK
   GUI/Console: Displays response
   TTS (optional): Speaks response
```

### Background Camera Loop

```
Separate Thread (daemon):
  While True:
    1. Capture frame from camera
    2. Publish to Redis: 'robot_camera' stream
    3. VisualCortex reads from Redis
    4. Runs object detection (OwlV2)
    5. Publishes results to Redis: 'detected_objects'
    6. Environment updates object memory
    7. Sleep 0.5s
    8. Repeat
```

## FastMCP Implementation

### Server Setup

**1. Initialize FastMCP**:
```python
from fastmcp import FastMCP

mcp = FastMCP("robot-environment")
```

**2. Define Tools**:
```python
@mcp.tool
def get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: List[float] = None,
    label: Optional[str] = None
) -> Optional[List[Dict]]:
    """
    Get list of detected objects with optional filters.

    Args:
        location: Spatial filter (left/right/above/below/close to)
        coordinate: Reference coordinate for location filter
        label: Filter by object label

    Returns:
        List of detected objects with positions and dimensions
    """
    detected_objects = env.get_detected_objects()
    objects = detected_objects.get_detected_objects_serializable(
        location, coordinate, label
    )
    return objects
```

**3. Run Server**:
```python
def main():
    # Initialize environment
    initialize_environment(
        el_api_key="",
        use_simulation=False,
        robot_id="niryo",
        verbose=True,
        start_camera_thread=True
    )

    # Start server
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
```

### Client Setup

**1. Initialize Client**:
```python
from fastmcp import Client
from fastmcp.client.transports import SSETransport

transport = SSETransport("http://127.0.0.1:8000/sse")
client = Client(transport)
```

**2. Connect and List Tools**:
```python
await client.__aenter__()
tools = await client.list_tools()
print(f"Available tools: {[t.name for t in tools]}")
```

**3. Call Tools**:
```python
result = await client.call_tool(
    "pick_place_object",
    {
        "object_name": "pencil",
        "pick_coordinate": [0.24, 0.02],
        "place_coordinate": [0.1, 0.11],
        "location": "right next to"
    }
)
```

**4. Integrate with Groq**:
```python
from groq import Groq

groq_client = Groq(api_key="your_key")

# Convert MCP tools to Groq format
groq_tools = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        }
    }
    for tool in tools
]

# LLM with tool calling
response = groq_client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[{"role": "user", "content": "Pick up the pencil"}],
    tools=groq_tools,
    tool_choice="auto"
)
```

## Coordinate Systems

### Three Systems Explained

**1. Image Coordinates (pixels)**:
```
┌─────────────────┐ (0, 0)
│                 │
│    640 x 480    │
│                 │
└─────────────────┘ (640, 480)
```

**2. Relative Coordinates (normalized)**:
```
(0.0, 0.0)
┌─────────────────┐
│                 │
│   Normalized    │
│    [0, 1]       │
│                 │
└─────────────────┘ (1.0, 1.0)
```

**3. World Coordinates (meters, robot base frame)**:
```
        Y (left)
        ↑
        │
        │
        └────→ X (forward)
       /
      / Z (up)
     ↙
  Robot Base
```

### Transformation Example

```python
# Object detected at pixel (320, 240) in 640x480 image

# Step 1: Pixel to relative
u_rel = 320 / 640 = 0.5
v_rel = 240 / 480 = 0.5

# Step 2: Relative to world (via workspace transformation)
pose = workspace.transform_camera2world_coords(
    workspace_id="niryo_ws",
    u_rel=0.5,
    v_rel=0.5,
    yaw=0.0  # Gripper rotation
)
# Result: x=0.25, y=0.0, z=0.01 (center of workspace)
```

### Niryo Coordinate Axes

- **X-axis**: Forward from robot base (toward workspace)
- **Y-axis**: Left-right (positive = left when facing robot)
- **Z-axis**: Vertical (positive = up)

### Location Semantics

```python
Location.LEFT_NEXT_TO   # y > reference_y (more left)
Location.RIGHT_NEXT_TO  # y < reference_y (more right)
Location.ABOVE          # x > reference_x (farther)
Location.BELOW          # x < reference_x (closer)
Location.CLOSE_TO       # distance <= 0.02m
```

## Object Detection

### Detection Pipeline

```
Camera Image
    ↓
OwlV2 Model (on GPU)
    ↓
Bounding Boxes + Confidence
    ↓
Filter by Confidence (> 0.15)
    ↓
Transform to World Coordinates
    ↓
Calculate Object Properties
    ↓
Create Object Instances
    ↓
Publish to Redis
```

### Object Properties

```python
obj = detected_objects[0]

# Identity
obj.label()              # "pencil"
obj.workspace().id()     # "niryo_ws"

# Position (world coordinates)
obj.x_com()              # 0.245 meters
obj.y_com()              # -0.053 meters

# Size (meters)
obj.width_m()            # 0.015 meters
obj.height_m()           # 0.120 meters
obj.size_m2()            # 0.0018 m² (18 cm²)

# Orientation
obj.gripper_rotation()   # 0.785 radians (45°)

# Bounding box (image coords)
obj.u_min(), obj.u_max() # 308, 322 pixels
obj.v_min(), obj.v_max() # 180, 300 pixels
```

### Detection Configuration

```python
# In robot_environment initialization
visual_cortex = VisualCortex(
    objdetect_model_id="owlv2",  # Detection model
    device="cuda",                # GPU acceleration
    config={
        'labels': ['pencil', 'cube', 'square', 'chocolate bar'],
        'confidence_threshold': 0.15,
        'iou_threshold': 0.5,
        'max_detections': 100
    }
)
```

## Advanced Usage

### Custom Tool Development

**Example: Create a "stack_objects" tool**:

```python
@mcp.tool
def stack_objects(
    bottom_object: str,
    top_object: str
) -> str:
    """
    Stack one object on top of another.

    Args:
        bottom_object: Label of base object
        top_object: Label of object to place on top

    Returns:
        Success message
    """
    # Get detected objects
    detected = env.get_detected_objects()

    # Find objects
    bottom = detected.get_detected_object_by_label(bottom_object)
    top = detected.get_detected_object_by_label(top_object)

    if not bottom or not top:
        return f"Could not find objects"

    # Execute stacking
    success = robot.pick_place_object(
        object_name=top.label(),
        pick_coordinate=[top.x_com(), top.y_com()],
        place_coordinate=[bottom.x_com(), bottom.y_com()],
        location=Location.ON_TOP_OF
    )

    return f"Stacked {top_object} on {bottom_object}"
```

### Batch Operations

```python
async def process_batch_commands(commands: List[str]):
    """Execute multiple commands sequentially."""
    client = RobotFastMCPClient(groq_api_key="key")
    await client.connect()

    results = []
    for cmd in commands:
        response = await client.chat(cmd)
        results.append(response)
        await asyncio.sleep(1)  # Pause between commands

    await client.disconnect()
    return results

# Usage
commands = [
    "Move to observation pose",
    "What objects do you see?",
    "Sort objects by size",
    "Create a triangle pattern"
]

results = asyncio.run(process_batch_commands(commands))
```

### Conditional Execution

```python
async def smart_placement():
    """Find best placement location automatically."""
    client = RobotFastMCPClient(groq_api_key="key")
    await client.connect()

    # LLM will use get_largest_free_space_with_center tool
    response = await client.chat(
        "Find the largest free space and place the pencil there"
    )

    await client.disconnect()
    return response
```

## Performance Tuning

### Detection Speed

**Fast Mode (YOLO-World)**:
```python
visual_cortex = VisualCortex(
    objdetect_model_id="yoloworld",  # Faster model
    device="cuda"
)
# ~10-25 FPS
```

**Accurate Mode (OwlV2)**:
```python
visual_cortex = VisualCortex(
    objdetect_model_id="owlv2",      # More accurate
    device="cuda"
)
# ~1-3 FPS
```

### Groq Model Selection

**For Speed**:
```python
client = RobotFastMCPClient(
    groq_api_key="key",
    model="llama-3.1-8b-instant"  # Very fast
)
```

**For Accuracy**:
```python
client = RobotFastMCPClient(
    groq_api_key="key",
    model="moonshotai/kimi-k2-instruct-0905"  # Better reasoning
)
```

### Camera Update Rate

```python
# In Environment initialization
env = Environment(
    ...,
    camera_update_rate=5  # FPS (adjust based on detection speed)
)
```

## Extending the System

### Adding a New Robot

**1. Create Controller**:
```python
# In robot_environment/robot/
class MyRobotController(RobotController):
    def __init__(self, robot_ip, verbose):
        super().__init__(robot_ip, verbose)
        # Initialize your robot's API

    def robot_pick_object(self, obj: Object) -> bool:
        # Implement pick logic
        pass

    def robot_place_object(self, pose: PoseObjectPNP) -> bool:
        # Implement place logic
        pass
```

**2. Add to Robot Selection**:
```python
# In robot_environment/robot/robot.py
if robot_id == "myrobot":
    self._robot_controller = MyRobotController(...)
```

### Adding New Object Labels

```python
@mcp.tool
def add_custom_object(label: str) -> str:
    """Add a new object type to recognition."""
    env.add_object_name2object_labels(label)
    return f"Added '{label}' to recognizable objects"

# Usage via LLM
"Please add 'screwdriver' to the list of recognizable objects"
```

### Custom Spatial Queries

```python
@mcp.tool
def get_objects_in_region(
    x_min: float, x_max: float,
    y_min: float, y_max: float
) -> List[Dict]:
    """Get all objects within a rectangular region."""
    detected = env.get_detected_objects()

    filtered = [
        obj for obj in detected
        if (x_min <= obj.x_com() <= x_max and
            y_min <= obj.y_com() <= y_max)
    ]

    return [obj.to_dict() for obj in filtered]
```

## Troubleshooting

See [docs/troubleshooting.md](troubleshooting.md) for common issues and solutions.

## Summary

The Robot MCP system provides:

✅ **Modern Architecture** - FastMCP with HTTP/SSE transport
✅ **Natural Language** - Groq LLM for command interpretation
✅ **Vision Integration** - Real-time object detection
✅ **Multi-Robot Support** - Hardware abstraction layer
✅ **Web Interface** - User-friendly Gradio GUI
✅ **Extensible Design** - Easy to add tools and robots
✅ **Production Ready** - Error handling and logging

For more information, see:
- [API Reference](api.md)
- [Examples](examples.md)
- [Troubleshooting](troubleshooting.md)

# MCP Robot Control - API Reference & Architecture

Complete API documentation and system architecture for the Robot MCP system with FastMCP and multi-LLM support.

## Table of Contents

- [System Architecture](#system-architecture)
- [API Tools Reference](#api-tools-reference)
- [Coordinate System](#coordinate-system)
- [Data Types](#data-types)
- [Integration Guide](#integration-guide)
- [Examples](#examples)

---

## System Architecture

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Client Layer                       │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ Universal    │  │ Gradio     │  │ Custom      │  │
│  │ LLM Client   │  │ Web GUI    │  │ Scripts     │  │
│  └──────────────┘  └────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                       │ HTTP/SSE
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

### Data Flow

**Complete Pick-and-Place Workflow:**

```
1. USER INPUT
   "Pick up the pencil and place it next to the red cube"
   ↓
2. LLM PROCESSING (OpenAI/Groq/Gemini/Ollama)
   ├─ Parses natural language
   ├─ Decides: Need to detect objects first
   └─ Generates tool call: get_detected_objects()
   ↓
3. FASTMCP CLIENT → SERVER (HTTP/SSE)
   Call: get_detected_objects()
   ↓
4. ROBOT ENVIRONMENT
   ├─ Query detected objects from memory
   └─ Return: [{label: "pencil", position: [0.15, -0.05]}, ...]
   ↓
5. LLM PROCESSES RESULT
   └─ Generates: pick_place_object(...) with coordinates
   ↓
6. ROBOT EXECUTION
   ├─ Move to observation pose
   ├─ Move to pick position
   ├─ Close gripper
   ├─ Move to place position
   ├─ Open gripper
   └─ Return to observation pose
   ↓
7. USER FEEDBACK
   "Done! Placed the pencil to the right of the red cube."
```

### Component Details

**FastMCP Server** (`server/fastmcp_robot_server.py`):
- Exposes 16 robot control tools
- Handles HTTP/SSE communication
- Manages robot environment lifecycle
- Converts tool calls to robot actions

**Universal Client** (`client/fastmcp_universal_client.py`):
- Supports 4 LLM providers (OpenAI, Groq, Gemini, Ollama)
- Auto-detects available APIs
- Manages conversation history
- Converts tool schemas to LLM format
- Handles streaming responses

**Robot Environment** (from `robot-environment` package):
- Hardware abstraction layer
- Vision-based object detection (OwlV2/YOLO-World)
- Coordinate transformations
- Motion planning
- Camera streaming via Redis

---

## API Tools Reference

### Robot Control Tools

#### pick_place_object

Complete pick-and-place operation in a single call.

**Function Signature:**
```python
pick_place_object(
    object_name: str,
    pick_coordinate: List[float],
    place_coordinate: List[float],
    location: Optional[Union[Location, str]] = None
) -> bool
```

**Parameters:**
- `object_name` (str): Object label (must match detection exactly, case-sensitive)
- `pick_coordinate` (List[float]): World coordinates [x, y] in meters
- `place_coordinate` (List[float]): Target coordinates [x, y] in meters
- `location` (Optional[str]): Relative placement position
  - `"left next to"` - Place to the left
  - `"right next to"` - Place to the right
  - `"above"` - Place above (farther in X)
  - `"below"` - Place below (closer in X)
  - `"on top of"` - Stack on top
  - `"inside"` - Place inside container
  - `"close to"` - Near coordinate
  - `None` - Exact coordinate

**Returns:** `True` on success

**Example:**
```python
# Via natural language
"Pick up the pencil at [0.15, -0.05] and place it right of the red cube at [0.20, 0.10]"

# Results in tool call
pick_place_object(
    object_name="pencil",
    pick_coordinate=[0.15, -0.05],
    place_coordinate=[0.20, 0.10],
    location="right next to"
)
```

**Notes:**
- Always call `get_detected_objects()` first to get current coordinates
- Object names are case-sensitive
- Robot automatically moves to observation pose before/after

---

#### pick_object

Pick up an object (without placing).

**Function Signature:**
```python
pick_object(
    object_name: str,
    pick_coordinate: List[float]
) -> bool
```

**Parameters:**
- `object_name` (str): Object label
- `pick_coordinate` (List[float]): World coordinates [x, y] in meters

**Returns:** `True` on success

**Example:**
```python
pick_object("pen", [0.18, -0.03])
```

**Notes:**
- Must be followed by `place_object()` to complete operation
- Gripper can hold objects up to ~5cm width

---

#### place_object

Place a currently held object.

**Function Signature:**
```python
place_object(
    place_coordinate: List[float],
    location: Optional[Union[Location, str]] = None
) -> bool
```

**Parameters:**
- `place_coordinate` (List[float]): Target coordinates [x, y] in meters
- `location` (Optional[str]): Relative placement (same options as `pick_place_object`)

**Returns:** `True` on success

**Example:**
```python
# First pick
pick_object("cube", [0.20, 0.05])
# Then place
place_object([0.18, -0.10], location="left next to")
```

---

#### push_object

Push an object (for items too large to grip).

**Function Signature:**
```python
push_object(
    object_name: str,
    push_coordinate: List[float],
    direction: str,
    distance: float
) -> bool
```

**Parameters:**
- `object_name` (str): Object label
- `push_coordinate` (List[float]): Current position [x, y] in meters
- `direction` (str): `"up"`, `"down"`, `"left"`, `"right"`
- `distance` (float): Distance in millimeters

**Returns:** `True` on success

**Example:**
```python
push_object("large box", [0.25, 0.05], "right", 50.0)
```

**Notes:**
- Use when object width > 5cm
- Direction is relative to robot's perspective

---

#### move2observation_pose

Move robot to observation position above workspace.

**Function Signature:**
```python
move2observation_pose(workspace_id: str) -> None
```

**Parameters:**
- `workspace_id` (str): Workspace ID (e.g., `"niryo_ws"`, `"gazebo_1"`)

**Returns:** None

**Example:**
```python
move2observation_pose("niryo_ws")
```

**Notes:**
- Called automatically before pick/place
- Positions camera for optimal object detection

---

### Object Detection Tools

#### get_detected_objects

Get list of all detected objects with optional filters.

**Function Signature:**
```python
get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: Optional[List[float]] = None,
    label: Optional[str] = None
) -> Optional[List[Dict]]
```

**Parameters:**
- `location` (str, optional): Spatial filter relative to `coordinate`
  - `"left next to"` - Objects to the left
  - `"right next to"` - Objects to the right
  - `"above"` - Objects above (farther in X)
  - `"below"` - Objects below (closer in X)
  - `"close to"` - Within 2cm radius
  - `None` - No filter (default)
- `coordinate` (List[float], optional): Reference coordinate [x, y]
- `label` (str, optional): Filter by object name

**Returns:** List of object dictionaries:
```python
[
    {
        "label": "pencil",
        "position": {"x": 0.150, "y": -0.050},
        "size": {
            "width_m": 0.015,
            "height_m": 0.120,
            "area_cm2": 18.0
        },
        "orientation_rad": 0.785
    },
    ...
]
```

**Examples:**
```python
# Get all objects
all_objects = get_detected_objects()

# Get objects near [0.2, 0.0]
nearby = get_detected_objects(
    location="close to",
    coordinate=[0.2, 0.0]
)

# Get all pencils
pencils = get_detected_objects(label="pencil")

# Get cubes to the left of [0.20, 0.0]
left_cubes = get_detected_objects(
    location="left next to",
    coordinate=[0.20, 0.0],
    label="cube"
)
```

---

#### get_detected_object

Find specific object at or near a coordinate.

**Function Signature:**
```python
get_detected_object(
    coordinate: List[float],
    label: Optional[str] = None
) -> Optional[Dict]
```

**Parameters:**
- `coordinate` (List[float]): Search coordinates [x, y]
- `label` (str, optional): Filter by object name

**Returns:** Single object dict or `None`

**Example:**
```python
# Find any object at [0.18, -0.05]
obj = get_detected_object([0.18, -0.05])

# Find specifically a "pen"
pen = get_detected_object([0.18, -0.05], label="pen")
```

**Notes:**
- Searches within 2cm radius
- Returns first match if multiple objects found

---

#### get_largest_detected_object

Get the largest object by area.

**Function Signature:**
```python
get_largest_detected_object() -> Tuple[Dict, float]
```

**Returns:** Tuple of (object_dict, size_in_m2)

**Example:**
```python
largest, size = get_largest_detected_object()
print(f"Largest: {largest['label']} at {size*10000:.1f} cm²")
```

---

#### get_smallest_detected_object

Get the smallest object by area.

**Function Signature:**
```python
get_smallest_detected_object() -> Tuple[Dict, float]
```

**Returns:** Tuple of (object_dict, size_in_m2)

**Example:**
```python
smallest, size = get_smallest_detected_object()
print(f"Smallest: {smallest['label']} ({size*10000:.1f} cm²)")
```

---

#### get_detected_objects_sorted

Get objects sorted by size.

**Function Signature:**
```python
get_detected_objects_sorted(
    ascending: bool = True
) -> List[Dict]
```

**Parameters:**
- `ascending` (bool): If True, smallest to largest; if False, largest to smallest

**Returns:** List of objects sorted by area

**Example:**
```python
# Smallest to largest
sorted_objs = get_detected_objects_sorted(ascending=True)

# Largest to smallest
sorted_objs = get_detected_objects_sorted(ascending=False)
```

---

### Workspace Tools

#### get_largest_free_space_with_center

Find largest empty space in workspace.

**Function Signature:**
```python
get_largest_free_space_with_center() -> Tuple[float, float, float]
```

**Returns:** Tuple of (area_m2, center_x, center_y)

**Example:**
```python
area, cx, cy = get_largest_free_space_with_center()
print(f"Free space: {area*10000:.1f} cm² at [{cx:.3f}, {cy:.3f}]")

# Use for safe placement
pick_place_object("cube", [0.15, -0.05], [cx, cy], None)
```

---

#### get_workspace_coordinate_from_point

Get coordinate of workspace corner or center.

**Function Signature:**
```python
get_workspace_coordinate_from_point(
    workspace_id: str,
    point: str
) -> Optional[List[float]]
```

**Parameters:**
- `workspace_id` (str): Workspace ID
- `point` (str): Point name
  - `"upper left corner"`
  - `"upper right corner"`
  - `"lower left corner"`
  - `"lower right corner"`
  - `"center point"`

**Returns:** Coordinate [x, y] in meters

**Example:**
```python
upper_left = get_workspace_coordinate_from_point("niryo_ws", "upper left corner")
center = get_workspace_coordinate_from_point("niryo_ws", "center point")
```

**Notes:**
- Niryo workspace: upper_left=[0.337, 0.087], lower_right=[0.163, -0.087]

---

#### get_object_labels_as_string

Get list of recognizable object types.

**Function Signature:**
```python
get_object_labels_as_string() -> str
```

**Returns:** Comma-separated string of object labels

**Example:**
```python
labels = get_object_labels_as_string()
# "pencil, pen, cube, cylinder, chocolate bar, ..."
```

---

#### add_object_name2object_labels

Add new object type to recognition system.

**Function Signature:**
```python
add_object_name2object_labels(object_name: str) -> str
```

**Parameters:**
- `object_name` (str): New object label

**Returns:** Confirmation message

**Example:**
```python
add_object_name2object_labels("screwdriver")
# Now can detect screwdrivers
```

---

### Feedback Tools

#### speak

Text-to-speech output for audio feedback.

**Function Signature:**
```python
speak(text: str) -> str
```

**Parameters:**
- `text` (str): Message to speak

**Returns:** Confirmation string

**Example:**
```python
speak("Task completed successfully")
```

**Notes:**
- Asynchronous - doesn't block execution
- Uses ElevenLabs or Kokoro TTS

---

## Coordinate System

### Robot Base Frame

```
        Y (left)
        ↑
        │
0.087 ──┼──────────── Upper workspace boundary
        │
    0 ──┼────────────  Center line (Y=0)
        │
-0.087 ─┼──────────── Lower workspace boundary
        │
        └────────────→ X (forward)
      0.163        0.337
     (closer)     (farther)
```

**Key Points:**
- **Origin:** Robot base
- **X-axis:** Forward/backward (values increase going forward)
- **Y-axis:** Left/right
  - Positive Y = left side (when facing robot)
  - Negative Y = right side
  - Y = 0 = center line
- **Z-axis:** Up/down (not used in 2D pick-and-place)
- **Units:** Meters

### Niryo Workspace Bounds

```python
X_MIN = 0.163  # Closer to robot
X_MAX = 0.337  # Farther from robot
Y_MIN = -0.087  # Right side
Y_MAX = 0.087   # Left side
```

### Coordinate Examples

```python
# Center of workspace
[0.25, 0.0]

# Upper left corner (far and left)
[0.337, 0.087]

# Lower right corner (close and right)
[0.163, -0.087]

# Left side, middle distance
[0.25, 0.06]

# Right side, far
[0.30, -0.05]
```

### Location Semantics

```python
Location.LEFT_NEXT_TO   # y > reference_y (positive direction)
Location.RIGHT_NEXT_TO  # y < reference_y (negative direction)
Location.ABOVE          # x > reference_x (farther from robot)
Location.BELOW          # x < reference_x (closer to robot)
Location.CLOSE_TO       # distance <= 0.02m (2cm radius)
```

---

## Data Types

### Object Dictionary

```python
{
    "label": str,           # Object name (e.g., "pencil")
    "position": {
        "x": float,         # X coordinate in meters
        "y": float          # Y coordinate in meters
    },
    "size": {
        "width_m": float,   # Width in meters
        "height_m": float,  # Height in meters
        "area_cm2": float   # Area in square centimeters
    },
    "orientation_rad": float  # Gripper rotation in radians
}
```

### Location Enum

```python
class Location(Enum):
    LEFT_NEXT_TO = "left next to"
    RIGHT_NEXT_TO = "right next to"
    ABOVE = "above"
    BELOW = "below"
    ON_TOP_OF = "on top of"
    INSIDE = "inside"
    CLOSE_TO = "close to"
    NONE = "none"
```

---

## Integration Guide

### Programmatic Usage

**Basic Integration:**

```python
from client.fastmcp_universal_client import RobotUniversalMCPClient
import asyncio

async def main():
    # Initialize client (auto-detects API)
    client = RobotUniversalMCPClient()

    # Or specify provider
    # client = RobotUniversalMCPClient(
    #     api_choice="openai",
    #     model="gpt-4o",
    #     temperature=0.7
    # )

    # Connect to server
    await client.connect()

    # Execute commands
    response1 = await client.chat("What objects do you see?")
    print(response1)

    response2 = await client.chat("Pick up the largest object")
    print(response2)

    response3 = await client.chat("Place it in the center")
    print(response3)

    # Disconnect
    await client.disconnect()

asyncio.run(main())
```

**Batch Processing:**

```python
async def batch_commands(commands: List[str]):
    client = RobotUniversalMCPClient()
    await client.connect()

    results = []
    for cmd in commands:
        response = await client.chat(cmd)
        results.append(response)
        await asyncio.sleep(1)

    await client.disconnect()
    return results

commands = [
    "What objects do you see?",
    "Sort objects by size",
    "Create a triangle pattern"
]
results = asyncio.run(batch_commands(commands))
```

**Custom System Prompt:**

```python
client = RobotUniversalMCPClient()
await client.connect()

# Modify system prompt
client.system_prompt = """You are a precision robot assistant.
Always verify coordinates before moving.
Speak aloud what you're doing.
If uncertain, ask for clarification."""

response = await client.chat("Organize the workspace")
```

**Provider Switching:**

```python
from client.llm_client import LLMClient

# Start with OpenAI
client = RobotUniversalMCPClient(api_choice="openai")
await client.connect()

# Switch to Groq mid-session
client.llm_client = LLMClient(
    api_choice="groq",
    model="moonshotai/kimi-k2-instruct-0905"
)

# Continue with new provider
response = await client.chat("Continue task")
```

---

## Examples

### Example 1: Workspace Scan

```python
async def scan_workspace():
    client = RobotUniversalMCPClient()
    await client.connect()

    response = await client.chat(
        "Scan the workspace and tell me: "
        "1. How many objects? "
        "2. What are they? "
        "3. Where are they located? "
        "4. Which is largest?"
    )

    print(response)
    await client.disconnect()
```

### Example 2: Sort by Size

```python
async def sort_objects():
    client = RobotUniversalMCPClient()
    await client.connect()

    response = await client.chat(
        "Sort all objects by size, placing them in a horizontal line "
        "from smallest to largest. Start at position [0.15, -0.05] "
        "and space them 8 centimeters apart."
    )

    print(response)
    await client.disconnect()
```

### Example 3: Pattern Creation

```python
async def create_triangle():
    client = RobotUniversalMCPClient()
    await client.connect()

    response = await client.chat(
        "Arrange all objects in a triangle pattern. "
        "Place the first object at [0.20, 0.0], "
        "second at [0.28, -0.06], "
        "and third at [0.28, 0.06]."
    )

    print(response)
    await client.disconnect()
```

### Example 4: Conditional Logic

```python
async def conditional_task():
    client = RobotUniversalMCPClient()
    await client.connect()

    response = await client.chat(
        "If there are more than 3 objects, arrange them in a 2x2 grid. "
        "Otherwise, arrange them in a straight line. "
        "Report the final arrangement."
    )

    print(response)
    await client.disconnect()
```

### Example 5: Multi-Provider Comparison

```python
async def compare_providers():
    providers = ["openai", "groq", "gemini"]
    command = "What objects do you see and where are they?"

    results = {}
    for provider in providers:
        try:
            client = RobotUniversalMCPClient(api_choice=provider)
            await client.connect()

            start = time.time()
            response = await client.chat(command)
            elapsed = time.time() - start

            results[provider] = {
                "response": response,
                "time": elapsed,
                "success": True
            }

            await client.disconnect()
        except Exception as e:
            results[provider] = {"error": str(e), "success": False}

    return results
```

---

## Performance Tips

### 1. Choose Right Model for Task

```python
# Complex reasoning - use best model
client = RobotUniversalMCPClient(
    api_choice="openai",
    model="gpt-4o"
)

# Simple tasks - use fast model
client = RobotUniversalMCPClient(
    api_choice="groq",
    model="llama-3.1-8b-instant"
)
```

### 2. Clear History Periodically

```python
# In interactive session
if len(client.conversation_history) > 20:
    client.conversation_history = client.conversation_history[-10:]
```

### 3. Batch Similar Operations

```python
# Instead of multiple separate commands
# "Move cube 1" -> "Move cube 2" -> "Move cube 3"

# Use single batch command
"Move all cubes to the left side, aligned in a row"
```

### 4. Use Direct Coordinates When Known

```python
# Faster - provides coordinates
"Pick up the pencil at [0.15, -0.05]"

# Slower - LLM must query detection
"Pick up the pencil" (requires get_detected_objects first)
```

---

## Error Handling

### Common Errors and Solutions

**Error:** `"Object not found"`
```python
# Solution: Verify detection first
objects = await client.chat("What objects do you see?")
# Then use exact label from detection
```

**Error:** `"Coordinates out of bounds"`
```python
# Solution: Check workspace bounds
# Valid: X=[0.163, 0.337], Y=[-0.087, 0.087]
```

**Error:** `"Tool execution failed"`
```python
# Solution: Check tool parameters
# Ensure types match (List[float] for coordinates, str for names)
```

**Error:** `"Maximum iterations reached"`
```python
# Solution: Break complex task into steps
# Instead of: "Do A, B, C, D, E"
# Do: "Do A and B" -> wait -> "Do C and D" -> wait -> "Do E"
```

---

For setup instructions and usage examples, see [Setup & Usage Guide](mcp_setup_guide.md).

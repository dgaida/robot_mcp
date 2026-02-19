# MCP Robot Control - API Reference & Architecture

Complete API documentation and system architecture for the Robot MCP system with FastMCP and multi-LLM support.

## Table of Contents

- [System Architecture](#system-architecture)
- [API Tools Reference](#api-tools-reference)
  - [Robot Control Tools](#robot-control-tools)
  - [Object Detection Tools](#object-detection-tools)
  - [Workspace Tools](#workspace-tools)
  - [Feedback Tools](#feedback-tools)
- [Coordinate System](#coordinate-system)
- [Data Types](#data-types)
- [Integration Guide](#integration-guide)
- [Error Handling](#error-handling)
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
│  │ (OpenAI/     │  │            │  │             │  │
│  │  Groq/       │  │            │  │             │  │
│  │  Gemini/     │  │            │  │             │  │
│  │  Ollama)     │  │            │  │             │  │
│  └──────────────┘  └────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                       │ HTTP/SSE
┌─────────────────────────────────────────────────────┐
│                  Server Layer                       │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ FastMCP      │  │ Tool       │  │ Robot       │  │
│  │ Server       │  │ Handlers   │  │ Environment │  │
│  │              │  │ (Pydantic  │  │             │  │
│  │              │  │ Validation)│  │             │  │
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

### Why FastMCP?

FastMCP offers several advantages as a production-ready framework for the Model Context Protocol:

1. **HTTP/SSE Transport**: More flexible than stdio, easier to debug
2. **Modern Python**: Uses async/await patterns with decorators
3. **Better Error Handling**: Clearer error messages and validation with Pydantic
4. **Easier Development**: Simpler tool registration with `@mcp.tool` decorators
5. **Network Ready**: Can run client and server on different machines
6. **Production Features**: Enterprise auth, deployment tools, testing frameworks

### Data Flow

**Complete Pick-and-Place Workflow:**

```
1. USER INPUT (Natural Language)
   "Pick up the pencil and place it next to the red cube"
   ↓
2. LLM PROCESSING (OpenAI/Groq/Gemini/Ollama)
   ├─ Chain-of-Thought: Explains task understanding
   ├─ Planning: Lists execution steps
   └─ Generates tool calls: get_detected_objects()
   ↓
3. FASTMCP CLIENT → SERVER (HTTP/SSE)
   Call: get_detected_objects()
   ↓
4. PYDANTIC VALIDATION
   ├─ Validates input parameters
   ├─ Checks coordinate format [x, y]
   └─ Verifies Location enum values
   ↓
5. ROBOT ENVIRONMENT
   ├─ Query detected objects from memory
   └─ Return: [{label: "pencil", position: [0.15, -0.05]}, ...]
   ↓
6. LLM PROCESSES RESULT
   └─ Generates: pick_place_object(...) with coordinates
   ↓
7. ROBOT EXECUTION
   ├─ Move to observation pose
   ├─ Move to pick position
   ├─ Close gripper
   ├─ Move to place position
   ├─ Open gripper
   └─ Return to observation pose
   ↓
8. USER FEEDBACK
   "Done! Placed the pencil to the right of the red cube."
```

### Component Details

**FastMCP Server** (`server/fastmcp_robot_server.py`):
- Exposes 16 robot control tools via `@mcp.tool` decorators
- Handles HTTP/SSE communication on port 8000
- Uses Pydantic models for input validation (see `server/schemas.py`)
- Manages robot environment lifecycle
- Converts tool calls to robot actions with error handling

**Universal Client** (`client/fastmcp_universal_client.py`):
- Supports 4 LLM providers via `LLMClient`:
  - OpenAI (GPT-4o, GPT-4o-mini)
  - Groq (Kimi K2, Llama 3.3, Mixtral)
  - Google Gemini (Gemini 2.0/2.5)
  - Ollama (Local models: llama3.2, mistral, etc.)
- Auto-detects available APIs based on environment keys
- Implements Chain-of-Thought prompting for transparency
- Manages conversation history with context limits
- Converts tool schemas to LLM function calling format
- Comprehensive logging to `log/mcp_client_*.log`

**Robot Environment** (from `robot-environment` package):
- Hardware abstraction layer for Niryo Ned2 and WidowX
- Vision-based object detection (OwlV2/YOLO-World)
- Coordinate transformations (image → world)
- Motion planning and collision avoidance
- Camera streaming via Redis (`robot_camera` stream)
- Object detection publishing (`detected_objects` stream)

---

## API Tools Reference

### Robot Control Tools

#### pick_place_object

Complete pick-and-place operation in a single call.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
@validate_input(PickPlaceInput)
def pick_place_object(
    object_name: str,
    pick_coordinate: List[float],
    place_coordinate: List[float],
    location: Optional[Union[Location, str]] = None
) -> str
```

**Pydantic Validation:**
```python
class PickPlaceInput(BaseModel):
    object_name: str = Field(..., min_length=1)
    pick_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    place_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    location: Optional[Union[Location, str]] = None
```

**Parameters:**
- `object_name` (str): Object label (must match detection exactly, case-sensitive)
  - Validated: Non-empty string
- `pick_coordinate` (List[float]): World coordinates [x, y] in meters
  - Validated: Exactly 2 numeric values
- `place_coordinate` (List[float]): Target coordinates [x, y] in meters
  - Validated: Exactly 2 numeric values
- `location` (Optional[str]): Relative placement position
  - Validated: Must be one of valid Location enum values
  - Options:
    - `"left next to"` - Place to the left
    - `"right next to"` - Place to the right
    - `"above"` - Place above (farther in X)
    - `"below"` - Place below (closer in X)
    - `"on top of"` - Stack on top
    - `"inside"` - Place inside container
    - `"close to"` - Near coordinate
    - `None` - Exact coordinate

**Returns:**
- Success: `"✓ Successfully picked 'pencil' from [0.150, -0.050] and placed it right next to coordinate [0.200, 0.100]"`
- Failure: `"❌ Failed to pick and place 'pencil'"` or validation error message

**Example:**
```python
# Via natural language (Chain-of-Thought)
User: "Pick up the pencil and place it right of the red cube"

🤖 CHAIN-OF-THOUGHT REASONING:
🎯 Task Understanding: Move pencil to position right of red cube
📋 Analysis: Need current positions of pencil and cube
🔧 Execution Plan:
   Step 1: get_detected_objects - Find all objects
   Step 2: pick_place_object - Move pencil with location="right next to"

# Results in validated tool call
pick_place_object(
    object_name="pencil",
    pick_coordinate=[0.15, -0.05],
    place_coordinate=[0.20, 0.10],
    location="right next to"
)
```

**Validation Errors:**
```python
# Invalid coordinate format
pick_place_object(
    object_name="pencil",
    pick_coordinate=[0.15],  # ❌ Only 1 value
    ...
)
→ "❌ Validation Error: pick_coordinate must have exactly 2 values"

# Invalid location
pick_place_object(
    ...,
    location="next_to"  # ❌ Invalid enum value
)
→ "❌ Validation Error: location must be one of: left next to, right next to, ..."
```

**Notes:**
- Always call `get_detected_objects()` first to get current coordinates
- Object names are case-sensitive
- Robot automatically moves to observation pose before/after
- All tool calls are logged to `log/mcp_server_*.log`

---

#### pick_object

Pick up an object (without placing).

**Function Signature:**
```python
@mcp.tool
@log_tool_call
@validate_input(PickObjectInput)
def pick_object(
    object_name: str,
    pick_coordinate: List[float]
) -> str
```

**Parameters:**
- `object_name` (str): Object label
- `pick_coordinate` (List[float]): World coordinates [x, y] in meters

**Returns:**
- Success: `"✓ Successfully picked 'pen' from [0.180, -0.030]"`
- Failure: `"❌ Failed to pick 'pen'"` or validation error

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
@mcp.tool
@log_tool_call
@validate_input(PlaceObjectInput)
def place_object(
    place_coordinate: List[float],
    location: Optional[Union[Location, str]] = None
) -> str
```

**Parameters:**
- `place_coordinate` (List[float]): Target coordinates [x, y] in meters
- `location` (Optional[str]): Relative placement (same options as `pick_place_object`)

**Returns:**
- Success: `"✓ Successfully placed object left next to coordinate [0.180, -0.100]"`
- Failure: `"❌ Failed to place object"` or validation error

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
@mcp.tool
@log_tool_call
@validate_input(PushObjectInput)
def push_object(
    object_name: str,
    push_coordinate: List[float],
    direction: str,
    distance: float
) -> str
```

**Pydantic Validation:**
```python
class PushObjectInput(BaseModel):
    object_name: str = Field(..., min_length=1)
    push_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    direction: str = Field(...)
    distance: float = Field(..., gt=0)

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        valid_directions = ["up", "down", "left", "right"]
        if v.lower() not in valid_directions:
            raise ValueError(f"Direction must be one of: {', '.join(valid_directions)}")
        return v
```

**Parameters:**
- `object_name` (str): Object label
- `push_coordinate` (List[float]): Current position [x, y] in meters
- `direction` (str): `"up"`, `"down"`, `"left"`, `"right"`
  - Validated: Must be one of the four directions
- `distance` (float): Distance in millimeters
  - Validated: Must be greater than 0

**Returns:**
- Success: `"✓ Successfully pushed 'large box' from [0.250, 0.050] right by 50.0mm"`
- Failure: `"❌ Failed to push 'large box'"` or validation error

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
@mcp.tool
@log_tool_call
def move2observation_pose(workspace_id: str) -> str
```

**Parameters:**
- `workspace_id` (str): Workspace ID (e.g., `"niryo_ws"`, `"gazebo_1"`)
  - Validated: Non-empty string

**Returns:**
- Success: `"✓ Moved to observation pose for workspace 'niryo_ws'"`
- Error: `"❌ Error moving to observation pose: ..."` or validation error

**Example:**
```python
move2observation_pose("niryo_ws")
```

**Notes:**
- Called automatically before pick/place
- Positions camera for optimal object detection

---

#### clear_collision_detected

Reset collision detection flag (Niryo only).

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def clear_collision_detected() -> str
```

**Returns:**
- Success: `"✓ Collision detection flag cleared"`
- Error: `"❌ Error clearing collision flag: ..."`

**Example:**
```python
clear_collision_detected()
```

**Notes:**
- Only needed after collision events
- Niryo-specific function

---

#### calibrate

Calibrate the robot.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def calibrate() -> str
```

**Returns:**
- Success: `"✓ Robot calibration completed successfully"`
- Failure: `"❌ Robot calibration failed"`

**Example:**
```python
calibrate()
```

---

### Object Detection Tools

#### get_detected_objects

Get list of all detected objects with optional filters.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
@validate_input(GetDetectedObjectsInput)
def get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: Optional[List[float]] = None,
    label: Optional[str] = None
) -> str
```

**Pydantic Validation:**
```python
class GetDetectedObjectsInput(BaseModel):
    location: Optional[Union[Location, str]] = None
    coordinate: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    label: Optional[str] = None
```

**Parameters:**
- `location` (str, optional): Spatial filter relative to `coordinate`
  - Validated: Must be valid Location enum value
  - Options:
    - `"left next to"` - Objects to the left
    - `"right next to"` - Objects to the right
    - `"above"` - Objects above (farther in X)
    - `"below"` - Objects below (closer in X)
    - `"close to"` - Within 2cm radius
    - `None` - No filter (default)
- `coordinate` (List[float], optional): Reference coordinate [x, y]
  - Validated: If provided, must be exactly 2 numeric values
- `label` (str, optional): Filter by object name

**Returns:**
- Success: JSON string with object list
  ```
  "✓ Found 3 object(s):
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
  ]"
  ```
- No objects: `"✓ No objects detected matching the criteria"`
- Error: `"❌ Error getting detected objects: ..."` or validation error

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

**Notes:**
- Always call this before pick/place to get current positions
- Object positions update continuously via camera
- Coordinates are center of mass (COM) of objects
- Robot automatically moves to observation pose first

---

#### get_detected_object

Find specific object at or near a coordinate.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def get_detected_object(
    coordinate: List[float],
    label: Optional[str] = None
) -> str
```

**Parameters:**
- `coordinate` (List[float]): World coordinates [x, y] to search near
  - Validated: Exactly 2 numeric values
- `label` (str, optional): Filter by object name

**Returns:**
- Success: JSON string with single object
  ```
  "✓ Found object near [0.180, -0.050]:
  {
    "label": "pen",
    "position": {"x": 0.180, "y": -0.050},
    ...
  }"
  ```
- Not found: `"✓ No object found near [0.180, -0.050]"`
- Error: `"❌ Error getting detected object: ..."` or validation error

**Example:**
```python
# Find any object at [0.18, -0.05]
obj = get_detected_object([0.18, -0.05])

# Find specifically a "pen" at that location
pen = get_detected_object([0.18, -0.05], label="pen")
```

**Notes:**
- Searches within 2cm radius of coordinate
- Returns first match if multiple objects found

---

#### get_largest_detected_object

Get the largest object by area.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def get_largest_detected_object() -> str
```

**Returns:**
- Success: JSON string with largest object and size
  ```
  "✓ Largest object (0.0025 m²):
  {
    "label": "blue square",
    "position": {"x": 0.180, "y": -0.100},
    "size": {"area_cm2": 25.0, ...}
  }"
  ```
- No objects: `"✓ No objects detected"`

**Example:**
```python
largest_obj = get_largest_detected_object()
```

---

#### get_smallest_detected_object

Get the smallest object by area.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def get_smallest_detected_object() -> str
```

**Returns:**
- Success: JSON string with smallest object and size
- No objects: `"✓ No objects detected"`

**Example:**
```python
smallest_obj = get_smallest_detected_object()
```

---

#### get_detected_objects_sorted

Get objects sorted by size.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def get_detected_objects_sorted(
    ascending: bool = True
) -> str
```

**Parameters:**
- `ascending` (bool): If True, smallest to largest; if False, largest to smallest
  - Validated: Must be boolean

**Returns:**
- Success: JSON string with sorted object list
  ```
  "✓ Found 3 object(s) sorted smallest to largest:
  [...]"
  ```
- No objects: `"✓ No objects detected"`

**Example:**
```python
# Smallest to largest
sorted_objs = get_detected_objects_sorted(ascending=True)

# Largest to smallest
sorted_objs = get_detected_objects_sorted(ascending=False)
```

**Notes:**
- Useful for size-based sorting tasks
- Sorting is by area (width × height)

---

### Workspace Tools

#### get_largest_free_space_with_center

Find largest empty space in workspace.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def get_largest_free_space_with_center() -> str
```

**Returns:**
- Success: `"✓ Largest free space: 0.0045 m² at center coordinates [0.240, -0.030]"`
- Error: `"❌ Error getting largest free space: ..."`

**Example:**
```python
result = get_largest_free_space_with_center()
# Use center coordinates for safe placement
```

**Notes:**
- Useful for finding safe placement locations
- Considers all detected objects as obstacles
- Returns center of largest contiguous free area

---

#### get_workspace_coordinate_from_point

Get coordinate of workspace corner or center.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
@validate_input(WorkspacePointInput)
def get_workspace_coordinate_from_point(
    workspace_id: str,
    point: str
) -> str
```

**Pydantic Validation:**
```python
class WorkspacePointInput(BaseModel):
    workspace_id: str = Field(..., min_length=1)
    point: str = Field(...)

    @field_validator("point")
    @classmethod
    def validate_point(cls, v):
        valid_points = [
            "upper left corner", "upper right corner",
            "lower left corner", "lower right corner", "center point"
        ]
        if v.lower() not in valid_points:
            raise ValueError(f"Point must be one of: {', '.join(valid_points)}")
        return v
```

**Parameters:**
- `workspace_id` (str): Workspace ID (e.g., `"niryo_ws"`)
- `point` (str): Point name
  - Validated: Must be one of valid point names
  - Options:
    - `"upper left corner"`
    - `"upper right corner"`
    - `"lower left corner"`
    - `"lower right corner"`
    - `"center point"`

**Returns:**
- Success: `"✓ Coordinate of 'center point' in workspace 'niryo_ws': [0.250, 0.000]"`
- Invalid: `"❌ Could not get coordinate for 'invalid_point' in workspace 'niryo_ws'"` or validation error

**Example:**
```python
upper_left = get_workspace_coordinate_from_point("niryo_ws", "upper left corner")
center = get_workspace_coordinate_from_point("niryo_ws", "center point")
```

**Notes:**
- Niryo workspace: upper_left=[0.337, 0.087], lower_right=[0.163, -0.087]
- Useful for boundary-aware placement
- Center is at approximately [0.25, 0.0]

---

#### get_object_labels_as_string

Get list of recognizable object types.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def get_object_labels_as_string() -> str
```

**Returns:**
- Success: `"✓ Detectable objects: pencil, pen, cube, cylinder, chocolate bar, cigarette, ..."`

**Example:**
```python
labels = get_object_labels_as_string()
```

**Notes:**
- Shows all labels the vision system can detect
- Labels are used in pick/place operations
- Case-sensitive matching required

---

#### add_object_name2object_labels

Add new object type to recognition system.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def add_object_name2object_labels(object_name: str) -> str
```

**Parameters:**
- `object_name` (str): New object label to add
  - Validated: Non-empty string

**Returns:**
- Success: `"✓ Added 'screwdriver' to the list of recognizable objects"`
- Error: `"❌ Validation Error: object_name must be a non-empty string"` or other error

**Example:**
```python
result = add_object_name2object_labels("screwdriver")
```

**Notes:**
- Extends detection capabilities dynamically
- New labels available immediately
- Vision model will attempt to detect new objects

---

### Feedback Tools

#### speak

Text-to-speech output for audio feedback.

**Function Signature:**
```python
@mcp.tool
@log_tool_call
def speak(text: str) -> str
```

**Parameters:**
- `text` (str): Message to speak
  - Validated: Non-empty string

**Returns:**
- Success: `"✓ Speaking: 'Task completed successfully'"`
- Error: `"❌ Validation Error: text must be a non-empty string"` or other error

**Example:**
```python
speak("I have picked up the pencil")
speak("Task completed successfully")
```

**Notes:**
- Asynchronous - doesn't block execution
- Uses ElevenLabs or Kokoro TTS (based on configuration)
- Useful for user feedback during long operations

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

### Object Dictionary (JSON Response)

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
from robot_workspace import Location

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

### Pydantic Input Models

All tool inputs are validated using Pydantic models in `server/schemas.py`:

```python
# Example: PickPlaceInput model
class PickPlaceInput(BaseModel):
    """Input validation for pick_place_object."""

    object_name: str = Field(..., min_length=1, description="Name of the object to pick")
    pick_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    place_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    location: Optional[Union[Location, str]] = Field(None, description="Relative placement location")

    class Config:
        arbitrary_types_allowed = True  # Allow enum types

    @field_validator("pick_coordinate", "place_coordinate")
    @classmethod
    def validate_coordinates(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numeric values [x, y]")
        return v

    @field_validator("location")
    @classmethod
    def validate_location(cls, v):
        if v is None:
            return v
        if isinstance(v, Location):
            return v
        if isinstance(v, str):
            valid_locations = [loc.value for loc in Location if loc
```

# TODO: example above uncomplete

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

### Advanced Integration Patterns

**Batch Processing:**

```python
async def batch_commands(commands: List[str]):
    """Execute multiple commands sequentially."""
    client = RobotUniversalMCPClient()
    await client.connect()

    results = []
    for cmd in commands:
        response = await client.chat(cmd)
        results.append(response)
        await asyncio.sleep(1)  # Pause between commands

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

# Modify system prompt for specific behavior
client.system_prompt = """You are a precision robot assistant.
Always verify coordinates before moving.
Speak aloud what you're doing.
If uncertain, ask for clarification.
CRITICAL: Never place objects where other objects exist."""

response = await client.chat("Organize the workspace")
```

**Provider Switching During Runtime:**

```python
# Start with OpenAI
client = RobotUniversalMCPClient(api_choice="openai")
await client.connect()

# Do some work
await client.chat("Pick up the pencil")

# Switch to Groq for faster inference
from llm_client import LLMClient
client.llm_client = LLMClient(
    api_choice="groq",
    model="llama-3.3-70b-versatile"
)

# Continue with new provider
await client.chat("Place it at the center")
```

**Error Handling:**

```python
async def safe_robot_command(command: str):
    """Execute command with comprehensive error handling."""
    client = RobotUniversalMCPClient()

    try:
        await client.connect()
        response = await client.chat(command)
        return {"success": True, "response": response}

    except ConnectionError as e:
        return {"success": False, "error": f"Connection failed: {e}"}

    except TimeoutError as e:
        return {"success": False, "error": f"Command timeout: {e}"}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}

    finally:
        try:
            await client.disconnect()
        except:
            pass

result = asyncio.run(safe_robot_command("What objects do you see?"))
if result["success"]:
    print(result["response"])
else:
    print(f"Error: {result['error']}")
```

### Direct FastMCP Client Usage

**Lower-Level Control:**

```python
from fastmcp import Client
from fastmcp.client.transports import SSETransport
import asyncio

async def direct_mcp_usage():
    """Use FastMCP client directly without LLM."""

    # Initialize transport and client
    transport = SSETransport("http://127.0.0.1:8000/sse")
    client = Client(transport)

    # Connect
    await client.__aenter__()

    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {[t.name for t in tools]}")

    # Call tool directly
    result = await client.call_tool(
        "get_detected_objects",
        {}
    )

    # Extract text from result
    if result.content:
        text = "\n".join([
            item.text for item in result.content
            if hasattr(item, "text")
        ])
        print(text)

    # Disconnect
    await client.__aexit__(None, None, None)

asyncio.run(direct_mcp_usage())
```

**Manual Tool Execution:**

```python
async def manual_pick_and_place():
    """Execute pick-and-place without LLM reasoning."""
    transport = SSETransport("http://127.0.0.1:8000/sse")
    client = Client(transport)

    await client.__aenter__()

    try:
        # Get detected objects
        objects_result = await client.call_tool(
            "get_detected_objects",
            {"label": "pencil"}
        )

        # Parse result (simplified)
        # In production, parse JSON properly
        print("Objects:", objects_result.content[0].text)

        # Execute pick and place
        pick_result = await client.call_tool(
            "pick_place_object",
            {
                "object_name": "pencil",
                "pick_coordinate": [0.15, -0.05],
                "place_coordinate": [0.20, 0.10],
                "location": "right next to"
            }
        )

        print("Pick-place result:", pick_result.content[0].text)

    finally:
        await client.__aexit__(None, None, None)

asyncio.run(manual_pick_and_place())
```

### Multi-Provider Comparison

**Benchmark Different LLM Providers:**

```python
import time
from typing import Dict, Any

async def compare_providers(task: str) -> Dict[str, Any]:
    """Compare performance across all available providers."""
    providers = ["openai", "groq", "gemini", "ollama"]
    results = {}

    for provider in providers:
        try:
            client = RobotUniversalMCPClient(api_choice=provider)
            await client.connect()

            start = time.time()
            response = await client.chat(task)
            elapsed = time.time() - start

            results[provider] = {
                "response": response,
                "time_seconds": elapsed,
                "model": client.llm_client.llm,
                "success": True
            }

            await client.disconnect()

        except Exception as e:
            results[provider] = {
                "error": str(e),
                "success": False
            }

    return results

# Usage
task = "What objects do you see and where are they?"
results = asyncio.run(compare_providers(task))

for provider, result in results.items():
    if result["success"]:
        print(f"\n{provider.upper()}: {result['time_seconds']:.2f}s")
        print(f"Model: {result['model']}")
        print(f"Response: {result['response'][:100]}...")
    else:
        print(f"\n{provider.upper()}: FAILED - {result['error']}")
```

### Conditional Execution

**Smart Task Execution:**

```python
async def smart_placement():
    """Find best placement location automatically."""
    client = RobotUniversalMCPClient()
    await client.connect()

    # LLM will use get_largest_free_space_with_center tool
    response = await client.chat(
        "Find the largest free space and place the pencil there"
    )

    await client.disconnect()
    return response

async def conditional_pickup():
    """Pick object with fallback if too large."""
    client = RobotUniversalMCPClient()
    await client.connect()

    response = await client.chat(
        "Try to pick up the large box. "
        "If it's too large for the gripper (width > 5cm), "
        "push it 50mm to the right instead."
    )

    await client.disconnect()
    return response
```

### State Management

**Stateful Robot Operations:**

```python
class RobotController:
    """Wrapper for stateful robot operations."""

    def __init__(self, api_choice: str = None):
        self.client = RobotUniversalMCPClient(api_choice=api_choice)
        self.connected = False
        self.task_history = []

    async def connect(self):
        """Connect to MCP server."""
        if not self.connected:
            await self.client.connect()
            self.connected = True

    async def execute_task(self, task: str) -> str:
        """Execute task and track history."""
        if not self.connected:
            await self.connect()

        response = await self.client.chat(task)

        self.task_history.append({
            "task": task,
            "response": response,
            "timestamp": time.time()
        })

        return response

    async def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state."""
        objects_response = await self.execute_task(
            "List all detected objects with their positions"
        )

        return {
            "objects": objects_response,
            "task_count": len(self.task_history),
            "last_task": self.task_history[-1] if self.task_history else None
        }

    async def cleanup(self):
        """Cleanup resources."""
        if self.connected:
            await self.client.disconnect()
            self.connected = False

# Usage
async def main():
    controller = RobotController(api_choice="groq")

    try:
        await controller.connect()

        # Execute multiple tasks
        await controller.execute_task("What objects do you see?")
        await controller.execute_task("Pick up the largest object")
        await controller.execute_task("Place it in the center")

        # Get final state
        state = await controller.get_workspace_state()
        print(f"Completed {state['task_count']} tasks")

    finally:
        await controller.cleanup()

asyncio.run(main())
```

### Integration with External Systems

**ROS Integration Example:**

```python
# Pseudo-code for ROS integration
import rospy
from std_msgs.msg import String

class ROSMCPBridge:
    """Bridge between ROS and MCP robot control."""

    def __init__(self):
        rospy.init_node('mcp_bridge')
        self.client = RobotUniversalMCPClient()

        # Subscribe to ROS command topic
        rospy.Subscriber('/robot/command', String, self.command_callback)

        # Publisher for results
        self.result_pub = rospy.Publisher('/robot/result', String, queue_size=10)

    async def command_callback(self, msg):
        """Handle incoming ROS commands."""
        command = msg.data

        # Execute via MCP
        response = await self.client.chat(command)

        # Publish result
        self.result_pub.publish(response)

    async def run(self):
        """Start the bridge."""
        await self.client.connect()
        rospy.spin()
        await self.client.disconnect()
```

**REST API Wrapper:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Global client (in production, use connection pooling)
robot_client = None

class RobotCommand(BaseModel):
    command: str
    provider: str = "auto"

@app.on_event("startup")
async def startup():
    global robot_client
    robot_client = RobotUniversalMCPClient()
    await robot_client.connect()

@app.on_event("shutdown")
async def shutdown():
    if robot_client:
        await robot_client.disconnect()

@app.post("/execute")
async def execute_command(cmd: RobotCommand):
    """Execute robot command via REST API."""
    try:
        response = await robot_client.chat(cmd.command)
        return {"success": True, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get robot status."""
    try:
        status = await robot_client.chat("What objects do you see?")
        return {"connected": True, "status": status}
    except:
        return {"connected": False}

# Run with: uvicorn script_name:app --reload
```

### Testing and Development

**Mock Client for Testing:**

```python
class MockMCPClient:
    """Mock client for testing without real robot."""

    def __init__(self):
        self.connected = False
        self.call_log = []

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def chat(self, message: str) -> str:
        """Return mock responses."""
        self.call_log.append(message)

        if "what objects" in message.lower():
            return "I can see a pencil at [0.15, -0.05] and a cube at [0.20, 0.10]"
        elif "pick" in message.lower():
            return "Successfully picked up the object"
        elif "place" in message.lower():
            return "Successfully placed the object"
        else:
            return "Command executed successfully"

# Use in tests
async def test_workflow():
    client = MockMCPClient()
    await client.connect()

    response1 = await client.chat("What objects do you see?")
    assert "pencil" in response1

    response2 = await client.chat("Pick up the pencil")
    assert "Successfully" in response2

    assert len(client.call_log) == 2

    await client.disconnect()
```

### Performance Optimization

**Connection Pooling:**

```python
from asyncio import Semaphore

class RobotClientPool:
    """Pool of robot clients for concurrent operations."""

    def __init__(self, size: int = 3):
        self.size = size
        self.clients = []
        self.semaphore = Semaphore(size)

    async def initialize(self):
        """Create client pool."""
        for i in range(self.size):
            client = RobotUniversalMCPClient()
            await client.connect()
            self.clients.append(client)

    async def execute(self, command: str) -> str:
        """Execute command using available client."""
        async with self.semaphore:
            # Get first available client
            client = self.clients[0]  # Simplified - use proper pooling
            return await client.chat(command)

    async def cleanup(self):
        """Close all clients."""
        for client in self.clients:
            await client.disconnect()

# Usage for high-throughput scenarios
async def batch_parallel():
    pool = RobotClientPool(size=3)
    await pool.initialize()

    commands = ["Command 1", "Command 2", "Command 3"]
    tasks = [pool.execute(cmd) for cmd in commands]
    results = await asyncio.gather(*tasks)

    await pool.cleanup()
    return results
```

### Logging and Monitoring

**Enhanced Logging:**

```python
import logging
from datetime import datetime

class LoggedRobotClient:
    """Robot client with comprehensive logging."""

    def __init__(self):
        self.client = RobotUniversalMCPClient()

        # Setup logging
        self.logger = logging.getLogger("RobotClient")
        handler = logging.FileHandler(
            f"robot_client_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def connect(self):
        self.logger.info("Connecting to MCP server...")
        await self.client.connect()
        self.logger.info("Connected successfully")

    async def execute(self, command: str) -> str:
        self.logger.info(f"Executing command: {command}")

        try:
            response = await self.client.chat(command)
            self.logger.info(f"Command succeeded: {response[:100]}...")
            return response
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            raise

    async def disconnect(self):
        self.logger.info("Disconnecting...")
        await self.client.disconnect()
        self.logger.info("Disconnected")
```

---

## Best Practices

### 1. Always Handle Connections Properly

```python
# ✅ Good - Use async context manager pattern
async def good_practice():
    client = RobotUniversalMCPClient()
    try:
        await client.connect()
        # Do work
        result = await client.chat("command")
    finally:
        await client.disconnect()

# ❌ Bad - No cleanup
async def bad_practice():
    client = RobotUniversalMCPClient()
    await client.connect()
    result = await client.chat("command")
    # Connection never closed!
```

### 2. Validate User Input

```python
# ✅ Good - Validate before sending
def validate_command(cmd: str) -> bool:
    if not cmd or not cmd.strip():
        return False
    if len(cmd) > 500:  # Too long
        return False
    return True

if validate_command(user_input):
    response = await client.chat(user_input)
```

### 3. Use Appropriate Provider for Task

```python
# Complex reasoning - use OpenAI GPT-4o
client = RobotUniversalMCPClient(
    api_choice="openai",
    model="gpt-4o"
)

# Simple tasks - use Groq (faster, free)
client = RobotUniversalMCPClient(
    api_choice="groq",
    model="llama-3.1-8b-instant"
)

# Offline/privacy - use Ollama
client = RobotUniversalMCPClient(
    api_choice="ollama",
    model="llama3.2:1b"
)
```

### 4. Monitor and Log

```python
# Always log important operations
logger.info(f"Starting task: {task_description}")
response = await client.chat(command)
logger.info(f"Task completed: {response}")
```

### 5. Handle Rate Limits

```python
import asyncio

async def rate_limited_execution(commands: List[str], delay: float = 2.0):
    """Execute commands with rate limiting."""
    results = []
    for cmd in commands:
        result = await client.chat(cmd)
        results.append(result)
        await asyncio.sleep(delay)  # Avoid rate limits
    return results
```

---

## Quick Reference

### Common Workflows

```python
# 1. Workspace scan
response = await client.chat("What objects do you see?")

# 2. Pick and place
response = await client.chat(
    "Pick up the pencil at [0.15, -0.05] and place it at [0.2, 0.1]"
)

# 3. Sort by size
response = await client.chat(
    "Sort all objects by size from smallest to largest"
)

# 4. Find safe placement
response = await client.chat(
    "Place the cube in the largest free space"
)

# 5. Conditional execution
response = await client.chat(
    "If the object is too large to pick, push it instead"
)
```

### Environment Setup

```python
# Load API keys
from dotenv import load_dotenv
load_dotenv("secrets.env")

# Initialize with specific provider
client = RobotUniversalMCPClient(
    api_choice="groq",  # or "openai", "gemini", "ollama"
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=4096
)
```

---

For more examples, see:
- [Setup Guide](mcp_setup_guide.md)
- [Examples](examples.md)
- [Troubleshooting](troubleshooting.md)

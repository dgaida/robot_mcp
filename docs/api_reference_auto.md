# Robot MCP - Auto-Generated API Reference

**Generated:** 2025-12-24 09:24:55  
**Source:** `server/fastmcp_robot_server_unified.py`  
**Total Tools:** 19

> ⚠️ **Note:** This documentation is auto-generated from source code.
> Do not edit manually. Run `python docs/generate_api_docs.py` to update.

---

## Table of Contents

- [Overview](#overview)
- [Quick Reference](#quick-reference)
- [API Tools](#api-tools)
  - [Feedback (2)](#tools-feedback)
  - [Object Detection (6)](#tools-object-detection)
  - [Robot Control (8)](#tools-robot-control)
  - [Workspace (3)](#tools-workspace)

## Overview

The Robot MCP system provides **19 tools** organized into **4 categories**:

| Category | Tools | Description |
|----------|-------|-------------|
| Feedback | 2 | User feedback via speech and text |
| Object Detection | 6 | Vision-based object recognition and querying |
| Robot Control | 8 | Physical robot manipulation and movement |
| Workspace | 3 | Workspace configuration and coordinate queries |

### Tool Categories

**Feedback** tools provide user communication:
- Text-to-speech output
- Task tracking for video overlays
- Status updates

**Object Detection** tools use computer vision:
- Real-time object detection via camera
- Spatial filtering (left/right/above/below)
- Size-based sorting and querying
- Label-based filtering

**Robot Control** tools provide physical manipulation capabilities:
- Pick and place operations
- Pushing large objects
- Precise movement control
- Calibration and safety

**Workspace** tools manage coordinate systems:
- Workspace boundary queries
- Free space detection
- Object label management
- Coordinate transformations

### Using the API

All tools are exposed via FastMCP and can be called through:

1. **Universal Client** - Multi-LLM support (OpenAI, Groq, Gemini, Ollama)
2. **Direct MCP Client** - Low-level FastMCP protocol
3. **Web GUI** - Gradio interface with voice input
4. **REST API** - HTTP endpoints (if enabled)

See [Setup Guide](mcp_setup_guide.md) for usage instructions.

## Quick Reference

### All Tools at a Glance

| Tool | Category | Input Validation | Description |
|------|----------|------------------|-------------|
| [`set_user_task`](#set_user_task) | Feedback | — | Set the current user task for display in video recordings. |
| [`speak`](#speak) | Feedback | — | Make the robot speak a message using text-to-speech. |
| [`get_detected_object`](#get_detected_object) | Object Detection | — | Retrieve a detected object at or near a specified world coor... |
| [`get_detected_objects`](#get_detected_objects) | Object Detection | ✓ | Get list of all objects detected by the camera in the worksp... |
| [`get_detected_objects_sorted`](#get_detected_objects_sorted) | Object Detection | — | Get detected objects sorted by size (area in square meters). |
| [`get_largest_detected_object`](#get_largest_detected_object) | Object Detection | — | Return the largest detected object based on its area in squa... |
| [`get_largest_free_space_with_center`](#get_largest_free_space_with_center) | Object Detection | — | Determine the largest free space in the workspace and its ce... |
| [`get_smallest_detected_object`](#get_smallest_detected_object) | Object Detection | — | Return the smallest detected object based on its area in squ... |
| [`calibrate`](#calibrate) | Robot Control | — | Calibrate the robot's joints for accurate movement. |
| [`clear_collision_detected`](#clear_collision_detected) | Robot Control | — | Reset the internal collision detection flag of the Niryo rob... |
| [`move2by`](#move2by) | Robot Control | — | Pick an object and move it a specified distance in a given d... |
| [`move2observation_pose`](#move2observation_pose) | Robot Control | — | Move robot to observation position above the specified works... |
| [`pick_object`](#pick_object) | Robot Control | ✓ | Pick up a specific object using the robot gripper. |
| [`pick_place_object`](#pick_place_object) | Robot Control | ✓ | Pick an object and place it at a target location in a single... |
| [`place_object`](#place_object) | Robot Control | ✓ | Place a previously picked object at the specified location. |
| [`push_object`](#push_object) | Robot Control | ✓ | Push a specific object to a new position using the robot gri... |
| [`add_object_name2object_labels`](#add_object_name2object_labels) | Workspace | — | Add a new object type to the list of recognizable objects. |
| [`get_object_labels_as_string`](#get_object_labels_as_string) | Workspace | — | Return all object labels that the detection model can recogn... |
| [`get_workspace_coordinate_from_point`](#get_workspace_coordinate_from_point) | Workspace | ✓ | Get the world coordinate of a special point in the workspace... |

## API Tools


---

### Tools: Feedback

#### set_user_task

**Category:** Feedback  

Set the current user task for display in video recordings.

**Signature:**
```python
def set_user_task(task: str) -> str
```

**Parameters:**

- `task` (`str`): The user's natural language task or command.

**Returns:**  
`str - str: Confirmation message that the task has been set.`

**Examples:**

```python
set_user_task("Pick up the red cube")
    # Video overlay shows: "Task: Pick up the red cube"

    set_user_task("Organize workspace by color")
    # Video overlay updates to show new task

    set_user_task("Sort objects by size")
    # Task text changes in video feed
```


---

#### speak

**Category:** Feedback  

Make the robot speak a message using text-to-speech.

**Signature:**
```python
def speak(text: str) -> str
```

**Parameters:**

- `text` (`str`): The message to speak. Must be a non-empty string.

**Returns:**  
`str - str: Confirmation message with the text being spoken, or error if failed.`

**Examples:**

```python
speak("I have picked up the pencil")
    # Robot says: "I have picked up the pencil"

    speak("Task completed successfully")
    # Robot announces task completion

    speak("Warning: Object detected in workspace")
    # Robot provides audio warning
```


---


---

### Tools: Object Detection

#### get_detected_object

**Category:** Object Detection  

Retrieve a detected object at or near a specified world coordinate.

**Signature:**
```python
def get_detected_object(coordinate: List[float], label: Optional[str] = None) -> str
```

**Parameters:**

- `coordinate` (`List[float]`): A 2D coordinate in the world coordinate system [x, y].

- `label` (`Optional[str]`, default: `None`)

**Returns:**  
`str - Object at [0.180, -0.050] (any object type)`

**Examples:**

```python
get_detected_object([0.18, -0.05])
    # Returns: Object at [0.180, -0.050] (any object type)

    get_detected_object([0.18, -0.05], label="pen")
    # Returns: Only a "pen" object at that location

    get_detected_object([0.25, 0.0])
    # Returns: Object near workspace center
```


---

#### get_detected_objects

**Category:** Object Detection  
**Validation:** `GetDetectedObjectsInput`  

Get list of all objects detected by the camera in the workspace.

**Signature:**
```python
def get_detected_objects(location: Union[Location, str] = Location.NONE, coordinate: Optional[List[float]] = None, label: Optional[str] = None) -> str
```

**Parameters:**

- `location` (`Union[Location, str]`, default: `Location.NONE`): Spatial filter relative to coordinate. Options:

- `coordinate` (`Optional[List[float]]`, default: `None`): Reference coordinate [x, y] in meters.

- `label` (`Optional[str]`, default: `None`)

**Returns:**  
`str - All detected objects in workspace`

**Examples:**

```python
get_detected_objects()
    # Returns: All detected objects in workspace

    get_detected_objects(location="close to", coordinate=[0.2, 0.0])
    # Returns: Objects within 2cm of [0.2, 0.0]

    get_detected_objects(label="pencil")
    # Returns: All objects labeled "pencil"

    get_detected_objects(location="left next to", coordinate=[0.20, 0.0], label="cube")
    # Returns: Cubes to the left of [0.20, 0.0]
```


---

#### get_detected_objects_sorted

**Category:** Object Detection  

Get detected objects sorted by size (area in square meters).

**Signature:**
```python
def get_detected_objects_sorted(ascending: bool = True) -> str
```

**Parameters:**

- `ascending` (`bool`, default: `True`): If True, sort smallest to largest. If False, sort

**Returns:**  
`str - [smallest object, medium object, largest object]`

**Examples:**

```python
get_detected_objects_sorted(ascending=True)
    # Returns: [smallest object, medium object, largest object]

    get_detected_objects_sorted(ascending=False)
    # Returns: [largest object, medium object, smallest object]

    get_detected_objects_sorted()
    # Returns: Objects sorted smallest to largest (default)
```


---

#### get_largest_detected_object

**Category:** Object Detection  

Return the largest detected object based on its area in square meters.

**Signature:**
```python
def get_largest_detected_object() -> str
```

**Returns:**  
`str - {"label": "blue square", "position": {...}, ...}`

**Examples:**

```python
get_largest_detected_object()
    # Returns: {"label": "blue square", "position": {...}, ...}

    # Use result for manipulation:
    # 1. largest = get_largest_detected_object()
    # 2. Parse JSON to get position
    # 3. pick_object(largest["label"], ...)
```


---

#### get_largest_free_space_with_center

**Category:** Object Detection  

Determine the largest free space in the workspace and its center coordinate.

**Signature:**
```python
def get_largest_free_space_with_center() -> str
```

**Returns:**  
`str - "0.0045 m² at center [0.240, -0.030]"`

**Examples:**

```python
get_largest_free_space_with_center()
    # Returns: "0.0045 m² at center [0.240, -0.030]"

    # Use for safe placement:
    # 1. result = get_largest_free_space_with_center()
    # 2. Parse to get center coordinates
    # 3. place_object(center_coords, None)

    # area, x, y = parse_result()
    # pick_place_object("cube", [0.2, 0.1], [x, y], None)
```


---

#### get_smallest_detected_object

**Category:** Object Detection  

Return the smallest detected object based on its area in square meters.

**Signature:**
```python
def get_smallest_detected_object() -> str
```

**Returns:**  
`str - {"label": "pen", "position": {...}, ...}`

**Examples:**

```python
get_smallest_detected_object()
    # Returns: {"label": "pen", "position": {...}, ...}

    # Use for precise picking:
    # 1. smallest = get_smallest_detected_object()
    # 2. Extract coordinates
    # 3. pick_object(smallest["label"], ...)
```


---


---

### Tools: Robot Control

#### calibrate

**Category:** Robot Control  

Calibrate the robot's joints for accurate movement.

**Signature:**
```python
def calibrate() -> str
```

**Returns:**  
`str - "✓ Robot calibration completed successfully"`

**Examples:**

```python
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
```


---

#### clear_collision_detected

**Category:** Robot Control  

Reset the internal collision detection flag of the Niryo robot.

**Signature:**
```python
def clear_collision_detected() -> str
```

**Returns:**  
`str - "✓ Collision detection flag cleared"`

**Examples:**

```python
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
```

**Notes:**

- Note: This is a Niryo-specific function. May not work with other robot types.

---

#### move2by

**Category:** Robot Control  

Pick an object and move it a specified distance in a given direction.

**Signature:**
```python
def move2by(object_name: str, pick_coordinate: List[float], direction: str, distance: float, z_offset: float = 0.001) -> str
```

**Parameters:**

- `object_name` (`str`): Name of object to move. Must match detection exactly.

- `pick_coordinate` (`List[float]`): Current world coordinates [x, y] in meters

- `direction` (`str`)

- `distance` (`float`)

- `z_offset` (`float`, default: `0.001`)

**Returns:**  
`str - str: Success message with pick and place coordinates, or error description.`

**Examples:**

```python
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
```


---

#### move2observation_pose

**Category:** Robot Control  

Move robot to observation position above the specified workspace.

**Signature:**
```python
def move2observation_pose(workspace_id: str) -> str
```

**Parameters:**

- `workspace_id` (`str`): ID of the workspace (e.g., "niryo_ws", "gazebo_1").

**Returns:**  
`str - str: Success message confirming movement to observation pose, or error description if movement failed.`

**Examples:**

```python
move2observation_pose("niryo_ws")
    # Moves to observation pose above niryo_ws workspace

    move2observation_pose("gazebo_1")
    # Moves to observation pose for simulation workspace

    # Typical workflow:
    # 1. move2observation_pose("niryo_ws")
    # 2. get_detected_objects()
    # 3. pick_place_object(...)
    # 4. move2observation_pose("niryo_ws")  # Return to home
```

**Notes:**

- Note: The robot automatically moves to observation pose before detection operations, but you may need to call this explicitly for other tasks.

---

#### pick_object

**Category:** Robot Control  
**Validation:** `PickObjectInput`  

Pick up a specific object using the robot gripper.

**Signature:**
```python
def pick_object(object_name: str, pick_coordinate: List[float], z_offset: float = 0.001) -> str
```

**Parameters:**

- `object_name` (`str`): Name of the object to pick. Ensure this name matches

- `pick_coordinate` (`List[float]`)

- `z_offset` (`float`, default: `0.001`)

**Returns:**  
`str - str: Success message with coordinates, or error description if failed.`

**Examples:**

```python
pick_object("pen", [0.01, -0.15])
    # Picks pen at world coordinates [0.01, -0.15]

    pick_object("cube", [0.20, 0.05])
    # Picks cube with default z_offset

    pick_object("pen", [0.01, -0.15], z_offset=0.02)
    # Picks pen with 2cm offset above detected position
    # Useful if pen is on top of another object
```

**Notes:**

- Note: Must be followed by place_object() to complete pick-and-place operation. For complete operation in one call, use pick_place_object() instead.

---

#### pick_place_object

**Category:** Robot Control  
**Validation:** `PickPlaceInput`  

Pick an object and place it at a target location in a single operation.

**Signature:**
```python
def pick_place_object(object_name: str, pick_coordinate: List[float], place_coordinate: List[float], location: Union[Location, str, None] = None, z_offset: float = 0.001) -> str
```

**Parameters:**

- `object_name` (`str`): Name of object to pick. Must match detection exactly.

- `pick_coordinate` (`List[float]`): World coordinates [x, y] in meters where

- `place_coordinate` (`List[float]`)

- `location` (`Union[Location, str, None]`, default: `None`)

- `z_offset` (`float`, default: `0.001`)

**Returns:**  
`str - str: Success message with pick and place coordinates, or error description.`

**Examples:**

```python
pick_place_object("chocolate bar", [-0.1, 0.01], [0.1, 0.11], "right next to")
    # Picks chocolate bar and places it right next to object at [0.1, 0.11]

    pick_place_object("cube", [0.2, 0.05], [0.3, 0.1], "on top of", z_offset=0.02)
    # Picks cube with 2cm z-offset and places it on top of target object

    pick_place_object("pen", [0.15, -0.05], [0.25, 0.0], None)
    # Picks pen and places at exact coordinates (no relative positioning)
```

**Notes:**

- Note: Always call get_detected_objects() first to get current object positions. Object names are case-sensitive and must match detection exactly.

---

#### place_object

**Category:** Robot Control  
**Validation:** `PlaceObjectInput`  

Place a previously picked object at the specified location.

**Signature:**
```python
def place_object(place_coordinate: List[float], location: Union[Location, str, None] = None) -> str
```

**Parameters:**

- `place_coordinate` (`List[float]`): Target coordinates [x, y] in meters where

- `location` (`Union[Location, str, None]`, default: `None`)

**Returns:**  
`str - str: Success message with placement coordinates and location, or error description if placement failed.`

**Examples:**

```python
place_object([0.2, 0.0], "left next to")
    # Places gripped object to the left of coordinate [0.2, 0.0]

    place_object([0.25, 0.05], "on top of")
    # Stacks gripped object on top of object at [0.25, 0.05]

    place_object([0.18, -0.10], None)
    # Places object at exact coordinates [0.18, -0.10]

    # Complete workflow:
    # pick_object("cube", [0.15, -0.05])
    # place_object([0.20, 0.10], "right next to")
```

**Notes:**

- Note: Must call pick_object() before calling this function. For complete pick-and-place in one operation, use pick_place_object() instead.

---

#### push_object

**Category:** Robot Control  
**Validation:** `PushObjectInput`  

Push a specific object to a new position using the robot gripper.

**Signature:**
```python
def push_object(object_name: str, push_coordinate: List[float], direction: str, distance: float) -> str
```

**Parameters:**

- `object_name` (`str`): Name of the object to push. Ensure the name matches

- `push_coordinate` (`List[float]`)

- `direction` (`str`)

- `distance` (`float`)

**Returns:**  
`str - str: Success message with object name, starting position, direction, and distance, or error description if push operation failed.`

**Examples:**

```python
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
```

**Notes:**

- Note: This is in millimeters, not meters (50.0 = 5cm).
- Note: Use push_object() instead of pick_place_object() when object width exceeds gripper capacity (~5cm). The robot approaches from the opposite side of the push direction to avoid collisions.

---


---

### Tools: Workspace

#### add_object_name2object_labels

**Category:** Workspace  

Add a new object type to the list of recognizable objects.

**Signature:**
```python
def add_object_name2object_labels(object_name: str) -> str
```

**Parameters:**

- `object_name` (`str`): Name of the new object to recognize. Should be descriptive

**Returns:**  
`str - "✓ Added 'screwdriver' to the list of recognizable objects" # Now the robot will look for screwdrivers in the workspace`

**Examples:**

```python
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
```

**Notes:**

- Note: The vision model's ability to detect the new object depends on its training. Well-known objects (tools, office supplies) are more likely to be detected.

---

#### get_object_labels_as_string

**Category:** Workspace  

Return all object labels that the detection model can recognize.

**Signature:**
```python
def get_object_labels_as_string() -> str
```

**Returns:**  
`str - "pencil, pen, cube, cylinder, chocolate bar, cigarette, ..."`

**Examples:**

```python
get_object_labels_as_string()
    # Returns: "pencil, pen, cube, cylinder, chocolate bar, cigarette, ..."

    # List available objects to user:
    # labels = get_object_labels_as_string()
    # print(f"I can detect: {labels}")
```

**Notes:**

- Note: Call this method when the user asks "What objects can you see?" or "What can you pick up?" to show detection capabilities.

---

#### get_workspace_coordinate_from_point

**Category:** Workspace  
**Validation:** `WorkspacePointInput`  

Get the world coordinate of a special point in the workspace.

**Signature:**
```python
def get_workspace_coordinate_from_point(workspace_id: str, point: str) -> str
```

**Parameters:**

- `workspace_id` (`str`): ID of the workspace (e.g., "niryo_ws", "gazebo_1").

- `point` (`str`)

**Returns:**  
`str - "✓ Coordinate of 'upper left corner': [0.337, 0.087]"`

**Examples:**

```python
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
```

**Notes:**

- Note: For Niryo workspace: - Upper left: [0.337, 0.087] (far and left) - Lower right: [0.163, -0.087] (close and right) - Center: ~[0.250, 0.000]

---


---

## Additional Resources

- **[Setup Guide](mcp_setup_guide.md)** - Installation and configuration
- **[Examples](examples.md)** - Common use cases and workflows
- **[Architecture](README.md)** - System design and data flow
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Contributing

To update this documentation:

1. Modify tool docstrings in `server/fastmcp_robot_server_unified.py`
2. Run: `python docs/generate_api_docs.py`
3. Commit both source and generated docs

## Validation

All tools with `@validate_input` decorator use Pydantic models for input validation.
See `server/schemas.py` for validation model definitions.

---

**Auto-generated by:** `docs/generate_api_docs.py`  
**Last updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

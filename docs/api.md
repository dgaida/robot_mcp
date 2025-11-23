# Robot MCP - API Reference

Complete documentation of all available MCP tools for robot control.

## Table of Contents

- [Robot Control Tools](#robot-control-tools)
- [Object Detection Tools](#object-detection-tools)
- [Workspace Tools](#workspace-tools)
- [Feedback Tools](#feedback-tools)
- [Data Types](#data-types)
- [Error Handling](#error-handling)

---

## Robot Control Tools

### pick_place_object

Complete pick-and-place operation in a single call.

**Signature:**
```python
pick_place_object(
    object_name: str,
    pick_coordinate: List[float],
    place_coordinate: List[float],
    location: Optional[Union[Location, str]] = None
) -> bool
```

**Parameters:**
- `object_name` (str): Name of the object to manipulate (must match detected object label exactly)
- `pick_coordinate` (List[float]): World coordinates [x, y] in meters where object is located
- `place_coordinate` (List[float]): World coordinates [x, y] in meters for placement
- `location` (Optional[str]): Relative placement position. Options:
  - `"left next to"` - Place to the left
  - `"right next to"` - Place to the right
  - `"above"` - Place above
  - `"below"` - Place below
  - `"on top of"` - Stack on top
  - `"inside"` - Place inside (for containers)
  - `"close to"` - Near the coordinate
  - `None` - Exact coordinate

**Returns:** `True` on success

**Example:**
```python
# Pick pencil and place it right of red cube
pick_place_object(
    object_name="pencil",
    pick_coordinate=[0.15, -0.05],
    place_coordinate=[0.20, 0.10],
    location="right next to"
)
```

**Notes:**
- Always call `get_detected_objects()` first to get current coordinates
- Object names are case-sensitive and must match exactly
- Coordinates are in robot base frame (meters)

---

### pick_object

Pick up an object and hold it in the gripper.

**Signature:**
```python
pick_object(
    object_name: str,
    pick_coordinate: List[float]
) -> bool
```

**Parameters:**
- `object_name` (str): Name of object to pick
- `pick_coordinate` (List[float]): World coordinates [x, y] in meters

**Returns:** `True` on success

**Example:**
```python
# Pick up a pen
pick_object(
    object_name="pen",
    pick_coordinate=[0.18, -0.03]
)
```

**Notes:**
- Must be followed by `place_object()` to complete operation
- Gripper can hold objects up to ~5cm width
- Robot will move to observation pose first

---

### place_object

Place a currently held object at target location.

**Signature:**
```python
place_object(
    place_coordinate: List[float],
    location: Optional[Union[Location, str]] = None
) -> bool
```

**Parameters:**
- `place_coordinate` (List[float]): Target world coordinates [x, y] in meters
- `location` (Optional[str]): Relative placement (same options as `pick_place_object`)

**Returns:** `True` on success

**Example:**
```python
# First pick an object
pick_object("cube", [0.20, 0.05])

# Then place it left of another object
place_object(
    place_coordinate=[0.18, -0.10],
    location="left next to"
)
```

**Notes:**
- Requires prior call to `pick_object()`
- Will fail if gripper is empty

---

### push_object

Push an object in a specified direction (for objects too large to grip).

**Signature:**
```python
push_object(
    object_name: str,
    push_coordinate: List[float],
    direction: str,
    distance: float
) -> bool
```

**Parameters:**
- `object_name` (str): Name of object to push
- `push_coordinate` (List[float]): Current object location [x, y] in meters
- `direction` (str): Push direction: `"up"`, `"down"`, `"left"`, `"right"`
- `distance` (float): Push distance in millimeters

**Returns:** `True` on success

**Example:**
```python
# Push large box 50mm to the right
push_object(
    object_name="large box",
    push_coordinate=[0.25, 0.05],
    direction="right",
    distance=50.0
)
```

**Notes:**
- Use when object width > 5cm (gripper limit)
- Direction is relative to robot's perspective
- Distance should be reasonable (10-100mm typically)

---

### move2observation_pose

Move robot to observation position above workspace.

**Signature:**
```python
move2observation_pose(workspace_id: str) -> None
```

**Parameters:**
- `workspace_id` (str): ID of target workspace (e.g., `"niryo_ws"`, `"gazebo_1"`)

**Returns:** None

**Example:**
```python
# Move to default workspace observation
move2observation_pose("niryo_ws")
```

**Notes:**
- Called automatically before pick/place operations
- Positions camera for optimal object detection
- Safe position for scanning workspace

---

### clear_collision_detected

Reset collision detection flag (Niryo only).

**Signature:**
```python
clear_collision_detected() -> None
```

**Returns:** None

**Example:**
```python
# After detecting collision
clear_collision_detected()
```

**Notes:**
- Only needed after collision events
- Niryo-specific function

---

## Object Detection Tools

### get_detected_objects

Get list of all detected objects with optional filters.

**Signature:**
```python
get_detected_objects(
    location: Union[Location, str] = Location.NONE,
    coordinate: Optional[List[float]] = None,
    label: Optional[str] = None
) -> Optional[List[Dict]]
```

**Parameters:**
- `location` (str, optional): Spatial filter relative to `coordinate`:
  - `"left next to"` - Objects to the left
  - `"right next to"` - Objects to the right
  - `"above"` - Objects above (farther in X)
  - `"below"` - Objects below (closer in X)
  - `"close to"` - Within 2cm radius
  - `None` - No filter (default)
- `coordinate` (List[float], optional): Reference coordinate for location filter
- `label` (str, optional): Filter by object name

**Returns:** List of detected objects, each containing:
```python
{
    "label": str,              # Object name
    "x": float,                # X coordinate (meters)
    "y": float,                # Y coordinate (meters)
    "width_m": float,          # Width in meters
    "height_m": float,         # Height in meters
    "area_m2": float,          # Area in square meters
    "rotation_rad": float,     # Gripper rotation (radians)
    "workspace_id": str        # Workspace ID
}
```

**Examples:**
```python
# Get all objects
all_objects = get_detected_objects()

# Get objects near coordinate [0.2, 0.0]
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

---

### get_detected_object

Find specific object at or near a coordinate.

**Signature:**
```python
get_detected_object(
    coordinate: List[float],
    label: Optional[str] = None
) -> Optional[Dict]
```

**Parameters:**
- `coordinate` (List[float]): World coordinates [x, y] to search near
- `label` (str, optional): Filter by object name

**Returns:** Single object dict (same format as `get_detected_objects`) or `None`

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
- Returns `None` if no object found

---

### get_largest_detected_object

Get the largest object by area.

**Signature:**
```python
get_largest_detected_object() -> Tuple[Dict, float]
```

**Returns:**
- Tuple of (object_dict, size_in_m2)

**Example:**
```python
largest_obj, size = get_largest_detected_object()
print(f"Largest: {largest_obj['label']} at {size*10000:.1f} cm²")
```

---

### get_smallest_detected_object

Get the smallest object by area.

**Signature:**
```python
get_smallest_detected_object() -> Tuple[Dict, float]
```

**Returns:**
- Tuple of (object_dict, size_in_m2)

**Example:**
```python
smallest_obj, size = get_smallest_detected_object()
print(f"Smallest: {smallest_obj['label']} ({size*10000:.1f} cm²)")
```

---

### get_detected_objects_sorted

Get objects sorted by size.

**Signature:**
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
# Get objects from smallest to largest
sorted_objs = get_detected_objects_sorted(ascending=True)

# Largest to smallest
sorted_objs = get_detected_objects_sorted(ascending=False)
```

**Notes:**
- Useful for size-based sorting tasks
- Sorting is by area (width × height)

---

## Workspace Tools

### get_largest_free_space_with_center

Find largest empty space in workspace.

**Signature:**
```python
get_largest_free_space_with_center() -> Tuple[float, float, float]
```

**Returns:**
- Tuple of (area_m2, center_x, center_y)
  - `area_m2`: Free space area in square meters
  - `center_x`: X coordinate of center (meters)
  - `center_y`: Y coordinate of center (meters)

**Example:**
```python
area, cx, cy = get_largest_free_space_with_center()
print(f"Largest free space: {area*10000:.1f} cm² at [{cx:.3f}, {cy:.3f}]")

# Use for safe placement
pick_place_object(
    object_name="cube",
    pick_coordinate=[0.15, -0.05],
    place_coordinate=[cx, cy],
    location=None
)
```

**Notes:**
- Useful for finding safe placement locations
- Considers all detected objects as obstacles
- Returns center of largest contiguous free area

---

### get_workspace_coordinate_from_point

Get coordinate of workspace corner or center.

**Signature:**
```python
get_workspace_coordinate_from_point(
    workspace_id: str,
    point: str
) -> Optional[List[float]]
```

**Parameters:**
- `workspace_id` (str): Workspace ID (e.g., `"niryo_ws"`)
- `point` (str): Point name:
  - `"upper left corner"`
  - `"upper right corner"`
  - `"lower left corner"`
  - `"lower right corner"`
  - `"center point"`

**Returns:** Coordinate [x, y] in meters, or None if invalid

**Example:**
```python
# Get workspace bounds
upper_left = get_workspace_coordinate_from_point("niryo_ws", "upper left corner")
lower_right = get_workspace_coordinate_from_point("niryo_ws", "lower right corner")
center = get_workspace_coordinate_from_point("niryo_ws", "center point")

print(f"Workspace from {upper_left} to {lower_right}")
```

**Notes:**
- Niryo workspace: upper_left=[0.337, 0.087], lower_right=[0.163, -0.087]
- Useful for boundary-aware placement
- Center is at approximately [0.25, 0.0]

---

### get_object_labels_as_string

Get list of recognizable object types.

**Signature:**
```python
get_object_labels_as_string() -> str
```

**Returns:** Comma-separated string of object labels

**Example:**
```python
labels = get_object_labels_as_string()
print(f"I can recognize: {labels}")
# Output: "pencil, pen, cube, cylinder, chocolate bar, cigarette, ..."
```

**Notes:**
- Shows all labels the vision system can detect
- Labels are used in pick/place operations
- Case-sensitive matching required

---

### add_object_name2object_labels

Add new object type to recognition system.

**Signature:**
```python
add_object_name2object_labels(object_name: str) -> str
```

**Parameters:**
- `object_name` (str): New object label to add

**Returns:** Confirmation message

**Example:**
```python
# Add custom object
result = add_object_name2object_labels("screwdriver")
print(result)  # "Added 'screwdriver' to recognizable objects"

# Now can detect and manipulate screwdrivers
```

**Notes:**
- Extends detection capabilities dynamically
- New labels available immediately
- Vision model will attempt to detect new objects

---

## Feedback Tools

### speak

Text-to-speech output for audio feedback.

**Signature:**
```python
speak(text: str) -> str
```

**Parameters:**
- `text` (str): Message to speak

**Returns:** Confirmation string

**Example:**
```python
speak("I have picked up the pencil")
speak("Task completed successfully")
```

**Notes:**
- Asynchronous - doesn't block execution
- Uses ElevenLabs or Kokoro TTS
- Useful for user feedback during long operations

---

## Data Types

### Location Enum

Relative placement positions:

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

### Coordinate System

**Robot Base Frame:**
- **Origin:** Robot base
- **X-axis:** Forward/backward (values increase forward)
- **Y-axis:** Left/right (positive = left, negative = right, 0 = center)
- **Z-axis:** Up/down (not used in 2D operations)
- **Units:** Meters

**Typical Niryo Workspace:**
```
        Y (left)
        ↑
        │
0.087 ──┼──────────── Upper boundary
        │
    0 ──┼────────────  Center line
        │
-0.087 ─┼──────────── Lower boundary
        │
        └────────────→ X (forward)
      0.163        0.337
```

---

## Error Handling

### Common Errors

**Object Not Found:**
```python
# Error: get_detected_objects() returns empty list
# Solution: Check camera view, ensure objects are visible
```

**Coordinates Out of Bounds:**
```python
# Error: Coordinates outside workspace
# Valid range: X=[0.163, 0.337], Y=[-0.087, 0.087]
# Solution: Use get_workspace_coordinate_from_point()
```

**Object Name Mismatch:**
```python
# Error: "pencil" vs "Pencil" - case sensitive!
# Solution: Use exact label from get_detected_objects()
```

**Gripper Too Small:**
```python
# Error: Object width > 5cm
# Solution: Use push_object() instead
```

### Best Practices

1. **Always detect before manipulating:**
   ```python
   objects = get_detected_objects()
   # Use coordinates from objects
   ```

2. **Verify object exists:**
   ```python
   obj = get_detected_object([x, y], label="target")
   if obj is None:
       speak("Object not found")
       return
   ```

3. **Use safe placement:**
   ```python
   area, cx, cy = get_largest_free_space_with_center()
   place_object([cx, cy])
   ```

4. **Handle collisions:**
   ```python
   try:
       pick_place_object(...)
   except CollisionError:
       clear_collision_detected()
   ```

---

## Quick Reference

### Most Common Workflow

```python
# 1. Scan workspace for pencils (your target object)
pencils = get_detected_objects(label="pencil")

# 2. Get first pencil as target object
target = pencils[0]

# 3. Find safe placement
area, cx, cy = get_largest_free_space_with_center()

# 4. Execute pick-and-place
pick_place_object(
    object_name="pencil",
    pick_coordinate=[target['x'], target['y']],
    place_coordinate=[cx, cy],
    location=None
)

# 5. Provide feedback
speak("Task completed")
```

### Tool Call Frequency Guidelines

- **High frequency:** `get_detected_objects()` - Call before each manipulation
- **Medium frequency:** `pick_place_object()` - At least once per session
- **Low frequency:** `get_largest_free_space_with_center()` - For safe placement
- **As needed:** `speak()` - For user feedback

---

For more information, see:
- [Main Documentation](../README.md)
- [Architecture Guide](mcp_api_reference.md#system-architecture)
- [Examples](examples.md)
- [Troubleshooting](troubleshooting.md)

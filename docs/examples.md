# Robot MCP - Examples

Common use cases and example workflows for robot control.

## Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Basic Operations](#basic-operations)
- [Object Manipulation](#object-manipulation)
- [Spatial Reasoning](#spatial-reasoning)
- [Complex Workflows](#complex-workflows)
- [Advanced Patterns](#advanced-patterns)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Quick Start Examples

### Example 1: Hello Robot

Simple first interaction with the robot.

```python
# Natural language
"What objects do you see?"

# The LLM will call:
# 1. move2observation_pose("niryo_ws")
# 2. get_detected_objects()
#
# Response: "I can see 3 objects: a pencil at [0.15, -0.05],
#            a red cube at [0.20, 0.10], and a pen at [0.18, -0.03]"
```

**Key Concepts:**
- Robot automatically moves to observation pose
- Object detection happens in background
- Coordinates are reported in meters

---

### Example 2: Simple Pick and Place

Move a single object.

```python
# Natural language
"Pick up the pencil and place it at [0.2, 0.1]"

# Executes:
pick_place_object(
    object_name="pencil",
    pick_coordinate=[0.15, -0.05],  # From detection
    place_coordinate=[0.2, 0.1],
    location=None
)
```

**Learning Points:**
- LLM detects objects first to get coordinates
- Object names must match detected labels
- Direct coordinate placement

---

### Example 3: Relative Placement

Place object relative to another.

```python
# Natural language
"Move the pencil to the right of the red cube"

# Executes:
pick_place_object(
    object_name="pencil",
    pick_coordinate=[0.15, -0.05],
    place_coordinate=[0.20, 0.10],  # Cube position
    location="right next to"
)
```

**Key Points:**
- `location` parameter defines relative position
- Robot calculates final placement automatically
- Maintains safe spacing between objects

---

## Basic Operations

### Workspace Scan

Comprehensive workspace analysis.

**Command:**
```python
"Scan the workspace and tell me everything you see"
```

**LLM Workflow:**
```python
# 1. Move to observation
move2observation_pose("niryo_ws")

# 2. Get all objects
objects = get_detected_objects()

# 3. Analyze and report
for obj in objects:
    print(f"{obj['label']} at [{obj['x']:.3f}, {obj['y']:.3f}]")
    print(f"  Size: {obj['width_m']*100:.1f}cm × {obj['height_m']*100:.1f}cm")
```

**Output:**
```
I can see 4 objects in the workspace:
1. Pencil at [0.15, -0.05] (1.5cm × 12cm)
2. Red cube at [0.20, 0.10] (4cm × 4cm)
3. Pen at [0.18, -0.03] (1.2cm × 14cm)
4. Blue square at [0.18, -0.10] (5cm × 5cm)
```

---

### Find Specific Objects

Query for particular items.

**Commands:**
```python
# Find by name
"Where is the pencil?"

# Find by properties
"Which object is the largest?"
"What's the smallest object?"

# Find by location
"What objects are on the left side?"
"Is there anything near [0.2, 0.0]?"
```

**Implementation Examples:**

```python
# Find by name
obj = get_detected_objects(label="pencil")[0]
# Returns first pencil found

# Find largest
largest, size = get_largest_detected_object()
# Returns biggest object and its area

# Find by location
left_objects = get_detected_objects(
    location="left next to",
    coordinate=[0.25, 0.0]  # Center reference
)
```

---

### Safe Placement

Find optimal placement location.

**Command:**
```python
"Find a safe place to put this object"
```

**Implementation:**
```python
# Get largest free space
area, cx, cy = get_largest_free_space_with_center()

# Place object there
place_object(
    place_coordinate=[cx, cy],
    location=None
)
```

**Use Cases:**
- Avoiding collisions
- Dense workspace organization
- Automated placement decisions

---

## Object Manipulation

### Sorting by Size

Arrange objects in size order.

**Command:**
```python
"Sort all objects by size from smallest to largest in a line"
```

**Workflow:**
```python
# 1. Get sorted objects
sorted_objs = get_detected_objects_sorted(ascending=True)

# 2. Calculate positions
start_x, start_y = 0.15, -0.05
spacing = 0.08  # 8cm apart

# 3. Move each object
for i, obj in enumerate(sorted_objs):
    target_y = start_y + (i * spacing)

    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=[start_x, target_y],
        location=None
    )
```

**Result:**
```
Smallest ─→ Largest
   │         │
  [│]       [██]
  pen      cube
```

---

### Color-Based Grouping

Organize by object attributes.

**Command:**
```python
"Group objects by color: red on left, blue on right"
```

**Workflow:**
```python
# 1. Detect all objects
all_objects = get_detected_objects()

# 2. Filter by color (from label)
red_objects = [obj for obj in all_objects if 'red' in obj['label']]
blue_objects = [obj for obj in all_objects if 'blue' in obj['label']]

# 3. Place red objects on left
for i, obj in enumerate(red_objects):
    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=[0.20, 0.06 + i*0.04],
        location=None
    )

# 4. Place blue objects on right
for i, obj in enumerate(blue_objects):
    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=[0.20, -0.06 - i*0.04],
        location=None
    )
```

---

### Stacking Objects

Create vertical arrangements.

**Command:**
```python
"Stack the small cube on top of the large cube"
```

**Implementation:**
```python
# Find both cubes
cubes = get_detected_objects(label="cube")
small_cube = min(cubes, key=lambda x: x['area_m2'])
large_cube = max(cubes, key=lambda x: x['area_m2'])

# Stack small on large
pick_place_object(
    object_name=small_cube['label'],
    pick_coordinate=[small_cube['x'], small_cube['y']],
    place_coordinate=[large_cube['x'], large_cube['y']],
    location="on top of"
)
```

**Notes:**
- Use `"on top of"` location
- Robot adjusts Z-height automatically
- Works for stable, flat objects

---

## Spatial Reasoning

### Pattern Creation

Geometric arrangements.

**Triangle Pattern:**
```python
"Arrange objects in a triangle"

# Workflow:
objects = get_detected_objects()

# Define triangle vertices
positions = [
    [0.20, 0.00],   # Top
    [0.28, -0.06],  # Bottom right
    [0.28, 0.06],   # Bottom left
]

# Place objects
for obj, pos in zip(objects[:3], positions):
    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=pos,
        location=None
    )
```

**Result:**
```
        ○
       / \
      /   \
     ○─────○
```

**Square Pattern:**
```python
positions = [
    [0.18, -0.06],  # Top-left
    [0.18, 0.06],   # Top-right
    [0.26, -0.06],  # Bottom-left
    [0.26, 0.06],   # Bottom-right
]
```

---

### Distance-Based Queries

Find objects by proximity.

**Command:**
```python
"What's the closest object to [0.2, 0.0]?"
```

**Implementation:**

```python
import math

def distance(obj, target):
    return math.sqrt(
        (obj['x'] - target[0])**2 +
        (obj['y'] - target[1])**2
    )

objects = get_detected_objects()
target = [0.2, 0.0]

closest = min(objects, key=lambda o: distance(o, target))
dist = distance(closest, target)

print(f"Closest: {closest['label']} at {dist*100:.1f}cm away")
```

There is an easier implementation:

```python
target = [0.2, 0.0]
object = get_detected_objects(location="close to", coordinate=target)

print(f"Closest: {closest['label']} at {dist*100:.1f}cm away")
```

---

### Boundary-Aware Placement

Respect workspace limits.

**Command:**
```python
"Place objects at the corners of the workspace"
```

**Implementation:**
```python
# Get workspace corners
upper_left = get_workspace_coordinate_from_point(
    "niryo_ws", "upper left corner"
)
upper_right = get_workspace_coordinate_from_point(
    "niryo_ws", "upper right corner"
)
lower_left = get_workspace_coordinate_from_point(
    "niryo_ws", "lower left corner"
)
lower_right = get_workspace_coordinate_from_point(
    "niryo_ws", "lower right corner"
)

corners = [upper_left, upper_right, lower_left, lower_right]
objects = get_detected_objects()[:4]

# Place at corners
for obj, corner in zip(objects, corners):
    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=corner,
        location=None
    )
```

---

## Complex Workflows

### Multi-Step Task

Sequential operations with dependencies.

**Command:**
```python
"""
Execute this sequence:
1. Find the pencil
2. Move it to the center
3. Find the largest object
4. Place it to the right of the pencil
5. Report final positions
"""
```

**LLM Reasoning:**
```python
# Step 1: Find pencil
objects = get_detected_objects()
pencil = get_detected_object([0, 0], label="pencil")

# Step 2: Move to center
center = get_workspace_coordinate_from_point("niryo_ws", "center point")
pick_place_object(
    object_name="pencil",
    pick_coordinate=[pencil['x'], pencil['y']],
    place_coordinate=center,
    location=None
)

# Step 3: Find largest
largest, _ = get_largest_detected_object()

# Step 4: Place right of pencil
pick_place_object(
    object_name=largest['label'],
    pick_coordinate=[largest['x'], largest['y']],
    place_coordinate=center,  # Pencil's new position
    location="right next to"
)

# Step 5: Report
final_objects = get_detected_objects()
speak("Task completed. All objects repositioned.")
```

---

### Conditional Logic

Decision-making based on workspace state.

**Command:**
```python
"""
If there are more than 3 objects, arrange them in a grid.
Otherwise, arrange them in a line.
"""
```

**Implementation:**
```python
objects = get_detected_objects()

if len(objects) > 3:
    # Grid arrangement (2x2)
    positions = [
        [0.18, -0.04], [0.18, 0.04],
        [0.26, -0.04], [0.26, 0.04],
    ]
else:
    # Line arrangement
    positions = [
        [0.20, -0.06],
        [0.20, 0.00],
        [0.20, 0.06],
    ]

# Execute placement
for obj, pos in zip(objects, positions):
    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=pos,
        location=None
    )
```

---

### Error Recovery

Graceful failure handling.

**Command:**
```python
"""
Try to pick the 'diamond'. If not found, pick any object instead.
"""
```

**Workflow:**
```python
# Try to find diamond
diamond = get_detected_objects(label="diamond")

if diamond is None:
    speak("Diamond not found. Picking alternative object.")
    # Get any object
    objects = get_detected_objects()
    if objects:
        target = objects[0]
    else:
        speak("No objects available")
        return
else:
    target = diamond[0]

# Proceed with pick
pick_place_object(
    object_name=target['label'],
    pick_coordinate=[target['x'], target['y']],
    place_coordinate=[0.25, 0.0],
    location=None
)
```

---

## Advanced Patterns

### Batch Processing

Process all objects with same operation.

**Command:**
```python
"Move all small objects (< 20 cm²) to the left side"
```

**Implementation:**
```python
objects = get_detected_objects()

# Filter by size
small_objects = [
    obj for obj in objects
    if obj['area_m2'] * 10000 < 20  # Convert to cm²
]

# Calculate positions on left side
left_x = 0.18
spacing = 0.05

for i, obj in enumerate(small_objects):
    target_y = -0.06 + (i * spacing)

    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=[left_x, target_y],
        location=None
    )

speak(f"Moved {len(small_objects)} small objects to the left")
```

---

### Push Operations

Handle oversized objects.

**Command:**
```python
"If the object is too large to pick, push it instead"
```

**Workflow:**
```python
objects = get_detected_objects()
target = get_largest_detected_object()[0]

# Check if grippable (< 5cm width)
if target['width_m'] > 0.05:
    speak("Object too large to grip. Using push operation.")

    # Push 5cm to the right
    push_object(
        object_name=target['label'],
        push_coordinate=[target['x'], target['y']],
        direction="right",
        distance=50.0  # millimeters
    )
else:
    # Normal pick-and-place
    pick_place_object(
        object_name=target['label'],
        pick_coordinate=[target['x'], target['y']],
        place_coordinate=[0.25, 0.0],
        location=None
    )
```

---

### Dynamic Workspace Adaptation

Adjust to changing conditions.

**Command:**
```python
"Organize the workspace: densely packed objects to the right, spread out objects to the left"
```

**Implementation:**
```python
objects = get_detected_objects()

# Calculate object density
def get_neighbors(obj, all_objs, radius=0.05):
    count = 0
    for other in all_objs:
        if other == obj:
            continue
        dist = math.sqrt(
            (obj['x'] - other['x'])**2 +
            (obj['y'] - other['y'])**2
        )
        if dist < radius:
            count += 1
    return count

# Classify objects
dense_objects = []
sparse_objects = []

for obj in objects:
    neighbors = get_neighbors(obj, objects)
    if neighbors >= 2:
        dense_objects.append(obj)
    else:
        sparse_objects.append(obj)

# Move dense to right, sparse to left
for i, obj in enumerate(dense_objects):
    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=[0.28, -0.06 + i*0.04],
        location=None
    )

for i, obj in enumerate(sparse_objects):
    pick_place_object(
        object_name=obj['label'],
        pick_coordinate=[obj['x'], obj['y']],
        place_coordinate=[0.18, -0.06 + i*0.04],
        location=None
    )
```

---

## Tips and Best Practices

Hint: Those tips should be followed by the LLM. But if something that the LLM does, goes wrong, you might find here the reasons for that.

### 1. Always Detect Before Manipulating

```python
# ✅ Good: Fresh detection
objects = get_detected_objects()
pencil = next(o for o in objects if o['label'] == 'pencil')
pick_object("pencil", [pencil['x'], pencil['y']])

# ❌ Bad: Stale coordinates
# Someone might have moved the object!
pick_object("pencil", [0.15, -0.05])  # Old position
```

### 2. Use Exact Label Matching

```python
# ✅ Good: Exact match
objects = get_detected_objects()
pencil = get_detected_object([0, 0], label="pencil")

# ❌ Bad: Case mismatch
pencil = get_detected_object([0, 0], label="Pencil")  # Won't match "pencil"
```

### 3. Check for Success

```python
# ✅ Good: Verify object exists
obj = get_detected_object([0.2, 0.0], label="target")
if obj is None:
    speak("Target object not found")
    return

pick_place_object(
    object_name="target",
    pick_coordinate=[obj['x'], obj['y']],
    place_coordinate=[0.25, 0.0],
    location=None
)

# ❌ Bad: No verification
pick_place_object("target", [0.2, 0.0], [0.25, 0.0], None)
```

### 4. Use Safe Placement

```python
# ✅ Good: Find free space
area, cx, cy = get_largest_free_space_with_center()
place_object([cx, cy], None)

# ⚠️ Risky: Fixed coordinates might collide
place_object([0.25, 0.0], None)
```

### 5. Provide User Feedback

```python
# ✅ Good: Informative
speak("Scanning workspace...")
objects = get_detected_objects()
speak(f"Found {len(objects)} objects")

# Process objects...
speak("Task completed successfully")

# ❌ Bad: Silent operation
objects = get_detected_objects()
# ... (user doesn't know what's happening)
```

### 6. Handle Workspace Bounds

```python
# ✅ Good: Respect limits
upper_left = get_workspace_coordinate_from_point("niryo_ws", "upper left corner")
lower_right = get_workspace_coordinate_from_point("niryo_ws", "lower right corner")

# Ensure coordinates are within bounds
x = max(lower_right[0], min(upper_left[0], target_x))
y = max(lower_right[1], min(upper_left[1], target_y))

place_object([x, y], None)

# ❌ Bad: Out of bounds
place_object([0.50, 0.20], None)  # Too far!
```

### 7. Use Relative Placement

```python
# ✅ Good: Adaptive positioning
pick_place_object(
    object_name="cube",
    pick_coordinate=[0.18, -0.05],
    place_coordinate=[0.22, 0.10],
    location="right next to"  # Robot calculates exact position
)

# ❌ Less flexible: Manual offset calculation
target_y = 0.10 - 0.04  # Manual spacing
pick_place_object("cube", [0.18, -0.05], [0.22, target_y], None)
```

---

## Running the Examples

### Using the Example Script

```bash
# Run specific example
python examples/fastmcp_main_client.py workspace_scan

# Run all examples
python examples/fastmcp_main_client.py all

# Custom model
python examples/fastmcp_main_client.py sort_by_size --model llama-3.1-8b-instant
```

### Interactive Mode

```bash
# Start interactive session
python client/fastmcp_groq_client.py

# Type commands naturally
You: Sort all objects by size
You: Arrange them in a triangle
You: Tell me what you see
```

### Web GUI

```bash
# Launch Gradio interface
./launch_gui.sh --robot niryo --real

# Use voice commands or text input
# See live camera feed
# Get visual feedback
```

---

For more information, see:
- [API Reference](api.md) - Complete tool documentation
- [Architecture Guide](README.md) - System design details
- [Troubleshooting](troubleshooting.md) - Common issues

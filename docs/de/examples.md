# Robot MCP - Examples

Common use cases and example workflows for robot control with natural language.

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

```
You: What objects do you see?

🔧 Calling tool: get_detected_objects
✓ Result: Detected 3 objects:
   - pencil at [0.15, -0.05]
   - red cube at [0.20, 0.10]
   - pen at [0.18, -0.03]

🤖 Assistant: I can see 3 objects: a pencil at [0.15, -0.05],
   a red cube at [0.20, 0.10], and a pen at [0.18, -0.03]
```

---

### Example 2: Simple Pick and Place

```
You: Pick up the pencil and place it at [0.2, 0.1]

🔧 Calling tool: get_detected_objects
✓ Result: Detected objects...

🔧 Calling tool: pick_place_object
   Arguments: {
     "object_name": "pencil",
     "pick_coordinate": [0.15, -0.05],
     "place_coordinate": [0.2, 0.1],
     "location": "none"
   }
✓ Result: Successfully picked and placed

🤖 Assistant: Done! I've picked up the pencil from [0.15, -0.05]
   and placed it at [0.2, 0.1].
```

---

### Example 3: Relative Placement

```
You: Move the pencil to the right of the red cube

🔧 Calling tool: get_detected_objects
✓ Result: Found pencil and red cube

🔧 Calling tool: pick_place_object
   Arguments: {
     "object_name": "pencil",
     "pick_coordinate": [0.15, -0.05],
     "place_coordinate": [0.20, 0.10],
     "location": "right next to"
   }
✓ Result: Successfully placed

🤖 Assistant: Perfect! I've moved the pencil to the right of
   the red cube.
```

---

## Basic Operations

### Workspace Scan

```
You: Scan the workspace and tell me everything you see

🔧 Calling tool: move2observation_pose("niryo_ws")
✓ Result: Moved to observation pose

🔧 Calling tool: get_detected_objects
✓ Result: Detected 4 objects...

🤖 Assistant: I can see 4 objects in the workspace:

   1. Pencil at [0.15, -0.05]
      Size: 1.5cm × 12cm (area: 18 cm²)

   2. Red cube at [0.20, 0.10]
      Size: 4cm × 4cm (area: 16 cm²)

   3. Pen at [0.18, -0.03]
      Size: 1.2cm × 14cm (area: 16.8 cm²)

   4. Blue square at [0.18, -0.10]
      Size: 5cm × 5cm (area: 25 cm²)
```

---

### Find Specific Objects

```
You: Where is the pencil?

🔧 Calling tool: get_detected_objects(label="pencil")
✓ Result: Found pencil at [0.15, -0.05]

🤖 Assistant: The pencil is located at coordinates [0.15, -0.05].
```

```
You: Which object is the largest?

🔧 Calling tool: get_largest_detected_object
✓ Result: {
  "label": "blue square",
  "position": {"x": 0.18, "y": -0.10},
  "size": {"area_cm2": 25.0}
}

🤖 Assistant: The largest object is the blue square at
   [0.18, -0.10] with an area of 25 square centimeters.
```

```
You: What objects are on the left side?

🔧 Calling tool: get_detected_objects(
     location="left next to",
     coordinate=[0.25, 0.0]
   )
✓ Result: Found 2 objects on left side

🤖 Assistant: On the left side I found:
   - Red cube at [0.20, 0.10]
   - Pen at [0.18, -0.03]
```

---

### Safe Placement

```
You: Find a safe place to put this object

🔧 Calling tool: get_largest_free_space_with_center
✓ Result: area=0.0050 m², center=[0.25, -0.04]

🔧 Calling tool: place_object([0.25, -0.04], None)
✓ Result: Successfully placed

🤖 Assistant: I've placed the object in the largest free space
   at [0.25, -0.04] (50 cm² available).
```

---

## Object Manipulation

### Sorting by Size

```
You: Sort all objects by size from smallest to largest in a line

🔧 Calling tool: get_detected_objects_sorted(ascending=True)
✓ Result: Sorted 3 objects by size

🔧 Calling tool: pick_place_object (object 1)
   Arguments: {"object_name": "pen", ...}
✓ Result: Placed smallest object

🔧 Calling tool: pick_place_object (object 2)
   Arguments: {"object_name": "pencil", ...}
✓ Result: Placed medium object

🔧 Calling tool: pick_place_object (object 3)
   Arguments: {"object_name": "cube", ...}
✓ Result: Placed largest object

🤖 Assistant: Done! I've sorted all objects by size in a horizontal
   line:
   - Smallest: pen at [0.15, -0.05]
   - Medium: pencil at [0.15, 0.03]
   - Largest: cube at [0.15, 0.11]
```

---

### Color-Based Grouping

```
You: Group objects by color: red on left, blue on right

🔧 Calling tool: get_detected_objects
✓ Result: Found 4 objects

🔧 Calling tool: pick_place_object (red cube)
✓ Result: Moved to left side

🔧 Calling tool: pick_place_object (blue square)
✓ Result: Moved to right side

🤖 Assistant: Objects are now grouped by color:
   - Left side (red): red cube at [0.20, 0.06]
   - Right side (blue): blue square at [0.20, -0.06]
```

---

### Stacking Objects

```
You: Stack the small cube on top of the large cube

🔧 Calling tool: get_detected_objects(label="cube")
✓ Result: Found 2 cubes

🔧 Calling tool: pick_place_object
   Arguments: {
     "object_name": "small cube",
     "pick_coordinate": [0.16, 0.05],
     "place_coordinate": [0.22, 0.10],
     "location": "on top of"
   }
✓ Result: Successfully stacked

🤖 Assistant: Perfect! I've stacked the small cube on top of
   the large cube at [0.22, 0.10].
```

---

## Spatial Reasoning

### Triangle Pattern

```
You: Arrange objects in a triangle

🔧 Calling tool: get_detected_objects
✓ Result: Found 3 objects

🔧 Calling tool: pick_place_object (object 1)
   → Top vertex: [0.20, 0.00]
✓ Result: Placed

🔧 Calling tool: pick_place_object (object 2)
   → Bottom right: [0.28, -0.06]
✓ Result: Placed

🔧 Calling tool: pick_place_object (object 3)
   → Bottom left: [0.28, 0.06]
✓ Result: Placed

🤖 Assistant: Done! Objects arranged in triangle:
        ○ [0.20, 0.00]
       / \
      /   \
     ○─────○
   [0.28,  [0.28,
    0.06]  -0.06]
```

---

### Distance-Based Queries

```
You: What's the closest object to [0.2, 0.0]?

🔧 Calling tool: get_detected_objects(
     location="close to",
     coordinate=[0.2, 0.0]
   )
✓ Result: Found pencil at [0.20, 0.02]

🤖 Assistant: The pencil at [0.20, 0.02] is closest,
   only 2 centimeters away from [0.2, 0.0].
```

---

### Boundary-Aware Placement

```
You: Place objects at the corners of the workspace

🔧 Calling tool: get_workspace_coordinate_from_point
   ("niryo_ws", "upper left corner")
✓ Result: [0.337, 0.087]

🔧 Calling tool: get_workspace_coordinate_from_point
   ("niryo_ws", "upper right corner")
✓ Result: [0.337, -0.087]

🔧 Calling tool: pick_place_object (4 times)
✓ Result: Placed all objects at corners

🤖 Assistant: Objects placed at workspace corners:
   - Upper left: [0.337, 0.087]
   - Upper right: [0.337, -0.087]
   - Lower left: [0.163, 0.087]
   - Lower right: [0.163, -0.087]
```

---

## Complex Workflows

### Multi-Step Task

```
You: Execute: 1) Find the pencil 2) Move it to center 3) Find
     largest object 4) Place it right of pencil 5) Report positions

🔧 Calling tool: get_detected_objects(label="pencil")
✓ Result: Pencil at [0.15, -0.05]

🔧 Calling tool: get_workspace_coordinate_from_point
   ("niryo_ws", "center point")
✓ Result: [0.25, 0.0]

🔧 Calling tool: pick_place_object
   → Move pencil to center
✓ Result: Placed at [0.25, 0.0]

🔧 Calling tool: get_largest_detected_object
✓ Result: Red cube at [0.20, 0.10]

🔧 Calling tool: pick_place_object
   Arguments: {
     "place_coordinate": [0.25, 0.0],
     "location": "right next to"
   }
✓ Result: Placed right of pencil

🔧 Calling tool: get_detected_objects
✓ Result: Updated positions

🤖 Assistant: Task completed! Final positions:
   - Pencil: [0.25, 0.0] (center)
   - Red cube: [0.25, -0.04] (right of pencil)
```

---

### Conditional Logic

```
You: If there are more than 3 objects, arrange in grid.
     Otherwise, arrange in line.

🔧 Calling tool: get_detected_objects
✓ Result: Found 4 objects

🤖 Assistant: Found 4 objects (more than 3), arranging in 2×2 grid...

🔧 Calling tool: pick_place_object (4 times)
   Grid positions:
   [0.18, -0.04] [0.18, 0.04]
   [0.26, -0.04] [0.26, 0.04]
✓ Result: All objects arranged

🤖 Assistant: Grid arrangement complete:
   ○  ○  [0.18, -0.04] [0.18, 0.04]
   ○  ○  [0.26, -0.04] [0.26, 0.04]
```

---

### Error Recovery

```
You: Try to pick the 'diamond'. If not found, pick any object instead.

🔧 Calling tool: get_detected_objects(label="diamond")
✓ Result: No object with label 'diamond' found

🔧 Calling tool: speak("Diamond not found. Picking alternative object.")
✓ Result: Speaking...

🔧 Calling tool: get_detected_objects
✓ Result: Found 3 objects

🔧 Calling tool: pick_object("pencil", [0.15, -0.05])
✓ Result: Picked pencil as alternative

🤖 Assistant: I couldn't find a diamond, so I picked up the
   pencil instead as an alternative.
```

---

## Advanced Patterns

### Batch Processing

```
You: Move all small objects (< 20 cm²) to the left side

🔧 Calling tool: get_detected_objects
✓ Result: Found 4 objects

🤖 Assistant: Filtering objects under 20 cm²...
   Found 2 small objects: pen (16.8 cm²), pencil (18 cm²)

🔧 Calling tool: pick_place_object (pen)
   → Target: [0.18, -0.06]
✓ Result: Placed

🔧 Calling tool: pick_place_object (pencil)
   → Target: [0.18, -0.01]
✓ Result: Placed

🔧 Calling tool: speak("Moved 2 small objects to the left")
✓ Result: Speaking...

🤖 Assistant: Done! Moved 2 small objects to the left side:
   - Pen: [0.18, -0.06]
   - Pencil: [0.18, -0.01]
```

---

### Push Operations

```
You: If the object is too large to pick, push it instead

🔧 Calling tool: get_largest_detected_object
✓ Result: Large box at [0.25, 0.05], width: 8cm

🤖 Assistant: Object width (8cm) exceeds gripper limit (5cm).
   Using push operation...

🔧 Calling tool: speak("Object too large to grip. Using push.")
✓ Result: Speaking...

🔧 Calling tool: push_object
   Arguments: {
     "object_name": "large box",
     "push_coordinate": [0.25, 0.05],
     "direction": "right",
     "distance": 50.0
   }
✓ Result: Pushed 50mm to the right

🤖 Assistant: Successfully pushed the large box 5cm to the right.
   New position: [0.25, 0.0]
```

---

### Dynamic Workspace Adaptation

```
You: Organize workspace: densely packed objects to right,
     spread out objects to left

🔧 Calling tool: get_detected_objects
✓ Result: Found 5 objects

🤖 Assistant: Analyzing object density...
   - Dense cluster (3 objects): right side
   - Sparse objects (2 objects): left side

🔧 Calling tool: pick_place_object (3 times for dense)
   → Right positions: [0.28, -0.08], [0.28, -0.04], [0.28, 0.0]
✓ Result: Dense objects positioned

🔧 Calling tool: pick_place_object (2 times for sparse)
   → Left positions: [0.18, -0.02], [0.18, 0.06]
✓ Result: Sparse objects positioned

🤖 Assistant: Workspace organized:
   - Left (sparse): 2 objects with 8cm spacing
   - Right (dense): 3 objects with 4cm spacing
```

---

## Tips and Best Practices

### 1. Always Detect Before Manipulating

```
✅ Good:
You: What objects do you see?
     [Wait for response]
You: Pick up the pencil

🔧 Uses fresh coordinates from detection

❌ Bad:
You: Pick up the pencil at [0.15, -0.05]

⚠️  Coordinates might be stale if object moved
```

---

### 2. Use Exact Label Matching

```
✅ Good:
🔧 get_detected_objects
✓ Result: Found "pencil" (lowercase)

🔧 pick_object("pencil", ...)  ← Exact match

❌ Bad:
🔧 pick_object("Pencil", ...)  ← Wrong case
🔧 pick_object("pen", ...)     ← Wrong object
```

---

### 3. Check for Success

```
✅ Good:
You: Pick up the pencil
     [Wait for confirmation]
You: Did that work?

🔧 Calling tool: get_detected_objects(label="pencil")
✓ Result: Pencil position changed ← Success!

❌ Bad:
You: Pick up pencil. Move cube. Stack items.
     [No verification between steps]
```

---

### 4. Use Safe Placement

```
✅ Good:
You: Place object in a safe location

🔧 get_largest_free_space_with_center
✓ Result: area=0.0045 m², center=[0.24, -0.03]

🔧 place_object([0.24, -0.03], None)
✓ Result: No collisions

❌ Risky:
You: Place object at [0.25, 0.0]

⚠️  Might collide with existing objects
```

---

### 5. Provide User Feedback

```
✅ Good:
🔧 speak("Scanning workspace...")
🔧 get_detected_objects
🔧 speak("Found 3 objects")
🔧 pick_place_object
🔧 speak("Task completed successfully")

❌ Bad:
🔧 get_detected_objects
🔧 pick_place_object
[Silent operation, user uncertain]
```

---

### 6. Handle Workspace Bounds

```
✅ Good:
🔧 get_workspace_coordinate_from_point("niryo_ws", "upper left")
✓ Result: [0.337, 0.087] ← Valid bounds

🔧 Ensure target within: X=[0.163, 0.337], Y=[-0.087, 0.087]

❌ Bad:
🔧 place_object([0.50, 0.20], None)
✗ Error: Coordinates out of workspace bounds
```

---

### 7. Use Relative Placement

```
✅ Good - Adaptive:
🔧 pick_place_object(
     ...,
     place_coordinate=[0.22, 0.10],
     location="right next to"
   )
✓ Robot calculates exact position with safe spacing

❌ Less Flexible - Manual:
target_y = 0.10 - 0.04  # Manual offset
🔧 pick_place_object(..., [0.22, 0.06], None)
⚠️  Fixed spacing might not be optimal
```

---

## Running the Examples

### Interactive Mode

```bash
python client/fastmcp_universal_client.py

You: Sort all objects by size
You: Arrange them in a triangle
You: Tell me what you see
```

---

### Single Command Mode

```bash
# Run one example
python client/fastmcp_universal_client.py \
  --command "Sort objects by size"

# Batch script
for cmd in "Scan workspace" "Sort by size" "Create triangle"; do
  python client/fastmcp_universal_client.py --command "$cmd"
  sleep 2
done
```

---

### Web GUI

```bash
python robot_gui/mcp_app.py --robot niryo

# Use voice commands or text input
# See live camera feed
# Get visual feedback
```

---

### Example Scripts

```bash
# Run specific example
python examples/universal_examples.py workspace_scan

# Run all examples
python examples/universal_examples.py all

# Compare LLM providers
python examples/universal_examples.py compare_providers
```

---

## Common Natural Language Commands

```
"What objects do you see?"
"Pick up the pencil"
"Move the red cube next to the blue square"
"Sort all objects by size"
"Arrange objects in a triangle"
"What's the largest object?"
"Place the smallest object in the center"
"Find a safe place to put this"
"Stack the small cube on the large cube"
"Push the large box 5cm to the right"
"Group objects by color"
"Is there anything near [0.2, 0.0]?"
```

---

For more information:
- [API Reference](api/index.md) - Complete tool documentation
- [Setup Guide](getting-started.md) - Installation and configuration
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

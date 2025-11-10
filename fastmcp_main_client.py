"""
Example Scripts for FastMCP Groq Client

Collection of example scripts demonstrating various robot control scenarios.

Usage:
    python examples_fastmcp_client.py [example_name]

Examples:
    python examples_fastmcp_client.py workspace_scan
    python examples_fastmcp_client.py sort_by_size
    python examples_fastmcp_client.py create_pattern
"""

import asyncio
import os
import sys
from client.fastmcp_groq_client import RobotFastMCPClient
from dotenv import load_dotenv


class RobotExamples:
    """Collection of robot control examples."""

    def __init__(self, api_key: str, model: str = "moonshotai/kimi-k2-instruct-0905"):
        self.client = RobotFastMCPClient(groq_api_key=api_key, model=model)

    async def connect(self):
        """Connect to the FastMCP server."""
        await self.client.connect()

    async def disconnect(self):
        """Disconnect from the FastMCP server."""
        await self.client.disconnect()

    async def run_example(self, example_func, description: str):
        """Run an example with proper formatting."""
        print("\n" + "=" * 70)
        print(f"🤖 EXAMPLE: {description}")
        print("=" * 70 + "\n")

        try:
            await example_func()
            print("\n" + "=" * 70)
            print("✓ Example completed successfully!")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback

            traceback.print_exc()

    # ==================== EXAMPLE 1: Workspace Scan ====================

    async def workspace_scan(self):
        """Scan the workspace and report all objects."""
        commands = [
            "What objects do you see in the workspace?",
            "Tell me the position and size of each object",
            "Which object is the largest?",
            "Which object is closest to the center at [0.2, 0.0]?",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(1)

    # ==================== EXAMPLE 2: Sort by Size ====================

    async def sort_by_size(self):
        """Sort all objects by size in a line."""
        response = await self.client.chat(
            "Sort all objects by size, placing them in a horizontal line "
            "from smallest to largest. Start at position [0.15, -0.05] "
            "and space them 8 centimeters apart."
        )
        print(f"🤖 Response: {response}")

    # ==================== EXAMPLE 3: Create Pattern ====================

    async def create_pattern(self):
        """Arrange objects in a geometric pattern."""
        response = await self.client.chat(
            "Arrange all objects in a triangle pattern. "
            "Place the first object at [0.15, 0.0], the second at [0.25, -0.08], "
            "and the third at [0.25, 0.08]."
        )
        print(f"🤖 Response: {response}")

    # ==================== EXAMPLE 4: Color Grouping ====================

    async def color_grouping(self):
        """Group objects by color."""
        commands = [
            "What objects do you see?",
            "Group all objects by color. Place red objects on the left side, "
            "blue objects in the middle, and other objects on the right.",
            "Tell me the final arrangement",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(1)

    # ==================== EXAMPLE 5: Object Swap ====================

    async def object_swap(self):
        """Swap positions of two objects."""
        response = await self.client.chat("Find the two largest objects and swap their positions")
        print(f"🤖 Response: {response}")

    # ==================== EXAMPLE 6: Boundary Check ====================

    async def boundary_placement(self):
        """Place objects at workspace boundaries."""
        commands = [
            "Get workspace information",
            "Place the smallest object at the top-left corner of the workspace",
            "Place the largest object at the bottom-right corner",
            "Place any remaining object in the center",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(1)

    # ==================== EXAMPLE 7: Relative Positioning ====================

    async def relative_positioning(self):
        """Practice relative object placement."""
        commands = [
            "Find a red object and a blue object",
            "Place the red object to the left of the blue object",
            "Now place a third object above the red object",
            "Describe the final arrangement",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(1)

    # ==================== EXAMPLE 8: Precision Placement ====================

    async def precision_placement(self):
        """Place object at precise coordinates."""
        response = await self.client.chat(
            "Pick up any object and place it exactly at coordinates [0.20, 0.05]. "
            "After placement, verify the object is at the correct position."
        )
        print(f"🤖 Response: {response}")

    # ==================== EXAMPLE 9: Multi-Step Task ====================

    async def multi_step_task(self):
        """Execute a complex multi-step task."""
        response = await self.client.chat(
            "Execute this sequence: "
            "1. Find all objects "
            "2. Move the smallest object to [0.15, 0.1] "
            "3. Move the largest object to the right of the smallest "
            "4. Move any remaining objects above the first two "
            "5. Report the final positions of all objects"
        )
        print(f"🤖 Response: {response}")

    # ==================== EXAMPLE 10: Conditional Logic ====================

    async def conditional_logic(self):
        """Use conditional logic in commands."""
        commands = [
            "If there's a pencil in the workspace, move it to [0.2, 0.0]. "
            "If not, tell me what objects are available.",
            "Check if there are more than 3 objects. "
            "If yes, arrange them in a square pattern. "
            "If no, arrange them in a line.",
            "Find the object nearest to [0.15, 0.0]. "
            "If it's within 5cm, move it to [0.25, 0.0]. "
            "Otherwise, just report its position.",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(2)

    # ==================== EXAMPLE 11: Push Operations ====================

    async def push_operations(self):
        """Demonstrate pushing objects."""
        commands = [
            "Find the largest object",
            "If the object is too large to pick (width > 8cm), push it 5cm to the right. "
            "Otherwise, pick and place it at [0.25, 0.0]",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(1)

    # ==================== EXAMPLE 12: Voice Feedback ====================

    async def voice_feedback(self):
        """Use text-to-speech for feedback."""
        commands = [
            "Scan the workspace and announce what you see",
            "Pick up the first object you found and announce what you're doing",
            "Place it at [0.2, 0.1] and announce completion",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(2)

    # ==================== EXAMPLE 13: Error Recovery ====================

    async def error_recovery(self):
        """Demonstrate error handling and recovery."""
        commands = [
            "Try to pick up an object called 'nonexistent_object'",
            "Since that failed, find any available object and pick it up",
            "Try to place it at coordinates [99.9, 99.9]",
            "Since that's out of bounds, place it at a safe position [0.2, 0.0]",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(1)

    # ==================== EXAMPLE 14: Batch Operations ====================

    async def batch_operations(self):
        """Perform batch operations on multiple objects."""
        response = await self.client.chat(
            "For each object in the workspace: "
            "1. Check its size "
            "2. If it's smaller than 20 square cm, move it to the left side [0.15, y] "
            "3. If it's larger, move it to the right side [0.25, y] "
            "4. Space them vertically by 8cm "
            "5. Report the final arrangement"
        )
        print(f"🤖 Response: {response}")

    # ==================== EXAMPLE 15: Workspace Cleanup ====================

    async def workspace_cleanup(self):
        """Organize a messy workspace."""
        commands = [
            "Analyze the current workspace",
            "Create an organized layout: "
            "- All cubes on the left "
            "- All cylinders in the middle "
            "- Everything else on the right "
            "- Align them in neat rows",
            "Verify the final organization",
        ]

        for cmd in commands:
            print(f"\n📝 Command: {cmd}")
            response = await self.client.chat(cmd)
            print(f"🤖 Response: {response}")
            await asyncio.sleep(2)


async def main():
    """Main entry point for examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robot Control Examples with FastMCP Groq Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Examples:
  workspace_scan      - Scan and analyze workspace
  sort_by_size        - Sort objects by size
  create_pattern      - Create geometric pattern
  color_grouping      - Group objects by color
  object_swap         - Swap two object positions
  boundary_placement  - Place objects at boundaries
  relative_positioning - Practice relative placement
  precision_placement - Precise coordinate placement
  multi_step_task     - Complex multi-step operations
  conditional_logic   - Use conditional commands
  push_operations     - Demonstrate pushing
  voice_feedback      - Use text-to-speech
  error_recovery      - Handle errors gracefully
  batch_operations    - Batch process objects
  workspace_cleanup   - Organize messy workspace
  all                 - Run all examples

Examples:
  python examples_fastmcp_client.py workspace_scan
  python examples_fastmcp_client.py sort_by_size
  python examples_fastmcp_client.py all
        """,
    )

    parser.add_argument(
        "example",
        nargs="?",
        default="workspace_scan",
        help="Example to run (default: workspace_scan)",
    )
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument(
        "--model",
        default="moonshotai/kimi-k2-instruct-0905",
        help="Groq model to use (default: moonshotai/kimi-k2-instruct-0905)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(dotenv_path="secrets.env")

    # Get API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key required.")
        print("Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Create examples instance
    examples = RobotExamples(api_key=api_key, model=args.model)

    # Map example names to methods
    example_map = {
        "workspace_scan": (examples.workspace_scan, "Workspace Scan"),
        "sort_by_size": (examples.sort_by_size, "Sort by Size"),
        "create_pattern": (examples.create_pattern, "Create Pattern"),
        "color_grouping": (examples.color_grouping, "Color Grouping"),
        "object_swap": (examples.object_swap, "Object Swap"),
        "boundary_placement": (examples.boundary_placement, "Boundary Placement"),
        "relative_positioning": (examples.relative_positioning, "Relative Positioning"),
        "precision_placement": (examples.precision_placement, "Precision Placement"),
        "multi_step_task": (examples.multi_step_task, "Multi-Step Task"),
        "conditional_logic": (examples.conditional_logic, "Conditional Logic"),
        "push_operations": (examples.push_operations, "Push Operations"),
        "voice_feedback": (examples.voice_feedback, "Voice Feedback"),
        "error_recovery": (examples.error_recovery, "Error Recovery"),
        "batch_operations": (examples.batch_operations, "Batch Operations"),
        "workspace_cleanup": (examples.workspace_cleanup, "Workspace Cleanup"),
    }

    try:
        await examples.connect()

        if args.example == "all":
            # Run all examples
            for name, (func, desc) in example_map.items():
                await examples.run_example(func, desc)
                await asyncio.sleep(3)  # Pause between examples
        elif args.example in example_map:
            # Run specific example
            func, desc = example_map[args.example]
            await examples.run_example(func, desc)
        else:
            print(f"Unknown example: {args.example}")
            print(f"Available examples: {', '.join(example_map.keys())}, all")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await examples.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

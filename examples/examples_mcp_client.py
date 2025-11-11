"""
Example Scripts for MCP Groq Client

Collection of example scripts demonstrating various robot control scenarios.

Usage:
    python examples/examples_mcp_client.py [example_name]

Examples:
    python examples/examples_mcp_client.py workspace_scan
    python examples/examples_mcp_client.py sort_by_size
    python examples/examples_mcp_client.py create_pattern
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.mcp_groq_client import RobotMCPClient
from examples.robot_examples_base import EXAMPLE_MAP, RobotExamplesBase


class RobotExamples(RobotExamplesBase):
    """Robot control examples using MCP client."""

    def __init__(self, api_key: str, robot_id: str = "niryo", use_simulation: bool = False):
        """
        Initialize with MCP client.

        Args:
            api_key: Groq API key
            robot_id: Robot type (niryo/widowx)
            use_simulation: Use simulation mode
        """
        client = RobotMCPClient(groq_api_key=api_key, robot_id=robot_id, use_simulation=use_simulation)
        super().__init__(client)
        self.api_key = api_key
        self.robot_id = robot_id
        self.use_simulation = use_simulation

    async def connect(self):
        """Connect to the MCP server."""
        await self.client.connect()

    async def disconnect(self):
        """Disconnect from the MCP server."""
        await self.client.disconnect()


async def main():
    """Main entry point for examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robot Control Examples with Groq MCP Client",
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
  python examples/examples_mcp_client.py workspace_scan
  python examples/examples_mcp_client.py sort_by_size --simulation
  python examples/examples_mcp_client.py all
        """,
    )

    parser.add_argument(
        "example",
        nargs="?",
        default="workspace_scan",
        help="Example to run (default: workspace_scan)",
    )
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--robot", choices=["niryo", "widowx"], default="niryo", help="Robot type (default: niryo)")
    parser.add_argument("--simulation", action="store_true", help="Use simulation mode")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key required.")
        print("Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Create examples instance
    examples = RobotExamples(api_key=api_key, robot_id=args.robot, use_simulation=args.simulation)

    try:
        await examples.connect()

        if args.example == "all":
            # Run all examples
            for name, description in EXAMPLE_MAP.items():
                example_func = getattr(examples, name)
                await examples.run_example(example_func, description)
                await asyncio.sleep(3)  # Pause between examples
        elif args.example in EXAMPLE_MAP:
            # Run specific example
            description = EXAMPLE_MAP[args.example]
            example_func = getattr(examples, args.example)
            await examples.run_example(example_func, description)
        else:
            print(f"Unknown example: {args.example}")
            print(f"Available examples: {', '.join(EXAMPLE_MAP.keys())}, all")
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

#!/usr/bin/env python3
"""
MCP Server Launcher with Configuration Management

This script helps launch and manage the MCP robot server with
different configurations and provides utilities for testing.

Usage:
    python mcp_server_launcher.py [command] [options]

Commands:
    run         - Run the MCP server
    test        - Test robot connection
    config      - Generate Claude Desktop configuration
    inspect     - Launch MCP inspector
    demo        - Run demo operations

Examples:
    python mcp_server_launcher.py run --robot niryo --simulation
    python mcp_server_launcher.py test
    python mcp_server_launcher.py config --output claude_config.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.absolute()


def generate_claude_config(
    robot_id: str = "niryo",
    use_simulation: bool = False,
    output_file: Optional[str] = None
) -> dict:
    """
    Generate Claude Desktop configuration for the MCP server.
    
    Args:
        robot_id: Robot type ("niryo" or "widowx")
        use_simulation: Whether to use simulation mode
        output_file: Optional file to write configuration to
        
    Returns:
        Configuration dictionary
    """
    project_root = get_project_root()
    server_path = project_root / "mcp_robot_server.py"
    
    config = {
        "mcpServers": {
            "robot-environment": {
                "command": "python",
                "args": [
                    str(server_path),
                    robot_id,
                    "true" if use_simulation else "false"
                ],
                "env": {
                    "PYTHONPATH": str(project_root)
                }
            }
        }
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration written to {output_file}")
        
        # Provide instructions
        if sys.platform == "darwin":
            config_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
        elif sys.platform == "win32":
            config_path = "%APPDATA%\\Claude\\claude_desktop_config.json"
        else:
            config_path = "~/.config/Claude/claude_desktop_config.json"
            
        print(f"\nTo use this configuration:")
        print(f"1. Copy the content to: {config_path}")
        print(f"2. Restart Claude Desktop")
        print(f"3. The robot-environment MCP server will be available")
    
    return config


async def test_robot_connection(
    robot_id: str = "niryo",
    use_simulation: bool = False
) -> bool:
    """
    Test robot connection and basic operations.
    
    Args:
        robot_id: Robot type
        use_simulation: Whether to use simulation
        
    Returns:
        True if test successful
    """
    print(f"Testing robot connection...")
    print(f"  Robot: {robot_id}")
    print(f"  Simulation: {use_simulation}")
    
    try:
        from robot_environment import Environment
        
        print("\n[1/4] Initializing environment...")
        env = Environment(
            el_api_key="",
            use_simulation=use_simulation,
            robot_id=robot_id,
            verbose=True
        )
        print("✓ Environment initialized")
        
        print("\n[2/4] Moving to observation pose...")
        workspace_id = env.get_workspace_home_id()
        env.robot_move2observation_pose(workspace_id)
        print(f"✓ Moved to observation pose for workspace '{workspace_id}'")
        
        print("\n[3/4] Getting robot pose...")
        pose = env.get_robot_pose()
        print(f"✓ Current pose: x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}")
        
        print("\n[4/4] Detecting objects...")
        await asyncio.sleep(2)  # Wait for detection
        objects = env.get_detected_objects()
        print(f"✓ Detected {len(objects)} object(s)")
        
        for obj in objects:
            print(f"  - {obj.label()} at [{obj.x_com():.3f}, {obj.y_com():.3f}]")
        
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_demo(
    robot_id: str = "niryo",
    use_simulation: bool = False
):
    """
    Run a demonstration of robot capabilities.
    
    Args:
        robot_id: Robot type
        use_simulation: Whether to use simulation
    """
    from robot_environment import Environment
    import time
    
    print("="*50)
    print("ROBOT DEMONSTRATION")
    print("="*50)
    
    # Initialize
    env = Environment(
        el_api_key="",
        use_simulation=use_simulation,
        robot_id=robot_id,
        verbose=True
    )
    
    robot = env.robot()
    
    # Demo 1: Workspace observation
    print("\n[Demo 1] Moving to observation pose...")
    env.robot_move2observation_pose(env.get_workspace_home_id())
    await asyncio.sleep(2)
    
    # Demo 2: Object detection
    print("\n[Demo 2] Detecting objects...")
    objects = env.get_detected_objects()
    print(f"Found {len(objects)} objects:")
    for i, obj in enumerate(objects, 1):
        print(f"  {i}. {obj.label()}")
        print(f"     Position: [{obj.x_com():.3f}, {obj.y_com():.3f}]")
        print(f"     Size: {obj.width_m():.3f}m × {obj.height_m():.3f}m")
    
    if len(objects) >= 2:
        # Demo 3: Pick and place
        print("\n[Demo 3] Pick and place operation...")
        obj1 = objects[0]
        obj2 = objects[1]
        
        print(f"Moving '{obj1.label()}' next to '{obj2.label()}'")
        
        success = robot.pick_place_object(
            object_name=obj1.label(),
            pick_coordinate=obj1.coordinate(),
            place_coordinate=obj2.coordinate(),
            location="right next to"
        )
        
        if success:
            print("✓ Pick and place completed")
            
            # Speak result
            env.oralcom_call_text2speech_async(
                f"I successfully moved the {obj1.label()} next to the {obj2.label()}"
            )
    else:
        print("\n[Demo 3] Skipped - need at least 2 objects")
    
    # Demo 4: Workspace info
    print("\n[Demo 4] Workspace information...")
    workspace = env.get_workspace(0)
    print(f"  ID: {workspace.id()}")
    print(f"  Size: {workspace.width_m():.3f}m × {workspace.height_m():.3f}m")
    print(f"  Center: [{workspace.xy_center_wc().x:.3f}, {workspace.xy_center_wc().y:.3f}]")
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50)


async def run_server(
    robot_id: str = "niryo",
    use_simulation: bool = False,
    verbose: bool = True
):
    """
    Run the MCP server.
    
    Args:
        robot_id: Robot type
        use_simulation: Whether to use simulation
        verbose: Enable verbose output
    """
    # Import the server
    # sys.path.insert(0, str(get_project_root()))
    from server.mcp_robot_server import RobotMCPServer

    # Start message goes to stderr (won't interfere with MCP)
    sys.stderr.write("=" * 50 + "\n")
    sys.stderr.write("STARTING MCP ROBOT SERVER\n")
    sys.stderr.write("=" * 50 + "\n")
    sys.stderr.write(f"Robot: {robot_id}\n")
    sys.stderr.write(f"Simulation: {use_simulation}\n")
    sys.stderr.write(f"Verbose: {verbose}\n")
    sys.stderr.write(f"Log file: mcp_server_*.log\n")
    sys.stderr.write("=" * 50 + "\n")
    sys.stderr.write("\nServer is running. Waiting for MCP client connection...\n")
    sys.stderr.write("(Check log file for details)\n")
    sys.stderr.write("(Use Ctrl+C to stop)\n\n")
    sys.stderr.flush()

    server = RobotMCPServer(
        el_api_key="",
        use_simulation=use_simulation,
        robot_id=robot_id,
        verbose=verbose
    )
    
    await server.run()


def launch_inspector(
    robot_id: str = "niryo",
    use_simulation: bool = False
):
    """
    Launch MCP Inspector for testing the server.
    
    Args:
        robot_id: Robot type
        use_simulation: Whether to use simulation
    """
    import subprocess
    
    server_path = get_project_root() / "mcp_robot_server.py"
    
    print("Launching MCP Inspector...")
    print(f"Server: {server_path}")
    print(f"Robot: {robot_id}, Simulation: {use_simulation}")
    
    cmd = [
        "npx",
        "@modelcontextprotocol/inspector",
        "python",
        str(server_path),
        robot_id,
        "true" if use_simulation else "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching inspector: {e}")
        print("\nMake sure you have Node.js installed:")
        print("  npm install -g @modelcontextprotocol/inspector")
    except FileNotFoundError:
        print("Error: npx not found. Please install Node.js first.")


def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description="MCP Robot Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run server with real Niryo robot
  python mcp_server_launcher.py run --robot niryo
  
  # Run server with simulated robot
  python mcp_server_launcher.py run --robot niryo --simulation
  
  # Test connection
  python mcp_server_launcher.py test --robot niryo
  
  # Generate config file
  python mcp_server_launcher.py config --output my_config.json
  
  # Run demo
  python mcp_server_launcher.py demo --robot niryo --simulation
  
  # Launch inspector
  python mcp_server_launcher.py inspect --robot niryo
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the MCP server")
    run_parser.add_argument(
        "--robot",
        choices=["niryo", "widowx"],
        default="niryo",
        help="Robot type (default: niryo)"
    )
    run_parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use simulation mode"
    )
    run_parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test robot connection")
    test_parser.add_argument(
        "--robot",
        choices=["niryo", "widowx"],
        default="niryo",
        help="Robot type (default: niryo)"
    )
    test_parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use simulation mode"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Generate Claude Desktop config")
    config_parser.add_argument(
        "--robot",
        choices=["niryo", "widowx"],
        default="niryo",
        help="Robot type (default: niryo)"
    )
    config_parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use simulation mode"
    )
    config_parser.add_argument(
        "--output",
        help="Output file path"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    demo_parser.add_argument(
        "--robot",
        choices=["niryo", "widowx"],
        default="niryo",
        help="Robot type (default: niryo)"
    )
    demo_parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use simulation mode"
    )
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Launch MCP inspector")
    inspect_parser.add_argument(
        "--robot",
        choices=["niryo", "widowx"],
        default="niryo",
        help="Robot type (default: niryo)"
    )
    inspect_parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use simulation mode"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "run":
        asyncio.run(run_server(
            robot_id=args.robot,
            use_simulation=args.simulation,
            verbose=not args.no_verbose
        ))
    
    elif args.command == "test":
        success = asyncio.run(test_robot_connection(
            robot_id=args.robot,
            use_simulation=args.simulation
        ))
        sys.exit(0 if success else 1)
    
    elif args.command == "config":
        generate_claude_config(
            robot_id=args.robot,
            use_simulation=args.simulation,
            output_file=args.output
        )
    
    elif args.command == "demo":
        asyncio.run(run_demo(
            robot_id=args.robot,
            use_simulation=args.simulation
        ))
    
    elif args.command == "inspect":
        launch_inspector(
            robot_id=args.robot,
            use_simulation=args.simulation
        )


if __name__ == "__main__":
    main()

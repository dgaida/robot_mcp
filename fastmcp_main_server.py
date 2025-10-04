#!/usr/bin/env python3
"""
FastMCP Robot Server Launcher

Simple launcher script for the FastMCP robot server.

Usage:
    python main_server.py [options]

Examples:
    python main_server.py
    python main_server.py --robot widowx
    python main_server.py --no-simulation
    python main_server.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys


def main():
    """Main entry point for the FastMCP server launcher."""
    parser = argparse.ArgumentParser(
        description="FastMCP Robot Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run server with default settings (Niryo, simulation, localhost:8000)
  python main_server.py
  
  # Run with WidowX robot
  python main_server.py --robot widowx
  
  # Run with real robot (no simulation)
  python main_server.py --no-simulation
  
  # Run on custom host/port
  python main_server.py --host 0.0.0.0 --port 8080
  
  # Combine options
  python main_server.py --robot widowx --no-simulation --port 9000
        """
    )
    
    parser.add_argument(
        "--robot",
        choices=["niryo", "widowx"],
        default="niryo",
        help="Robot type (default: niryo)"
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="Use real robot instead of simulation"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Don't start camera thread"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("STARTING FASTMCP ROBOT SERVER")
    print("=" * 60)
    print(f"Robot:        {args.robot}")
    print(f"Simulation:   {not args.no_simulation}")
    print(f"Host:         {args.host}")
    print(f"Port:         {args.port}")
    print(f"Camera:       {not args.no_camera}")
    print(f"Verbose:      {args.verbose}")
    print("=" * 60)
    print(f"\nServer will be available at: http://{args.host}:{args.port}")
    print("SSE endpoint: http://{args.host}:{args.port}/sse")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Import and configure the server module
    try:
        # Modify the server module's environment settings before import
        import server.fastmcp_robot_server as server_module
        
        # Reconfigure the environment with command line arguments
        from robot_environment import Environment
        
        server_module.env = Environment(
            el_api_key="",
            use_simulation=not args.no_simulation,
            robot_id=args.robot,
            verbose=args.verbose,
            start_camera_thread=not args.no_camera
        )
        server_module.robot = server_module.env.robot()
        
        # Run the server with specified host and port
        server_module.mcp.run(
            transport="sse",
            host=args.host,
            port=args.port
        )
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("SERVER STOPPED")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

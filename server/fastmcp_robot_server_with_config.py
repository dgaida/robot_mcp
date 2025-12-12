# server/fastmcp_robot_server_with_config.py
"""
FastMCP Robot Server with Centralized Configuration Management

This version uses ConfigManager for all settings instead of scattered
command-line arguments and environment variables.

Usage:
    # Use default config
    python server/fastmcp_robot_server_with_config.py

    # Specify config file
    python server/fastmcp_robot_server_with_config.py --config path/to/config.yaml

    # Use environment-specific config
    ROBOT_ENV=production python server/fastmcp_robot_server_with_config.py

    # Override specific settings
    python server/fastmcp_robot_server_with_config.py --robot-type widowx --port 8080
"""

import argparse
import functools
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Union

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
from pydantic import ValidationError
from robot_environment import Environment
from robot_workspace import Location

from config.config_manager import ConfigManager, load_config
from server.schemas import (
    PickPlaceInput,
)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

config: ConfigManager = None
env: Environment = None
robot = None
logger: logging.Logger = None


# ============================================================================
# LOGGING SETUP WITH CONFIG
# ============================================================================


def setup_logging(config: ConfigManager) -> logging.Logger:
    """Setup logging based on configuration."""

    # Create log directory
    log_dir = Path(config.server.log_dir)
    log_dir.mkdir(exist_ok=True)

    # Log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mcp_server_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=config.server.log_level,
        format=config.logging.format,
        datefmt=config.logging.date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    # Set module-specific log levels
    for module_name, level in config.logging.levels.items():
        logging.getLogger(module_name).setLevel(level)

    logger = logging.getLogger("FastMCPRobotServer")

    logger.info("=" * 80)
    logger.info(f"FastMCP Robot Server starting - Log file: {log_file}")
    logger.info(f"Configuration loaded from: {config}")
    logger.info("=" * 80)

    return logger


# ============================================================================
# VALIDATION DECORATOR
# ============================================================================


def validate_input(model_class):
    """Decorator to validate tool inputs using Pydantic models."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                validated_data = model_class(**kwargs)
                result = func(*args, **validated_data.model_dump())
                return result
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error["loc"])
                    msg = error["msg"]
                    errors.append(f"{field}: {msg}")

                error_msg = f"❌ Validation Error in {func.__name__}:\n" + "\n".join(f"  • {err}" for err in errors)
                logger.error(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"❌ Error in {func.__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg

        return wrapper

    return decorator


def log_tool_call(func):
    """Decorator to log all tool calls with parameters and results."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__

        logger.info("-" * 60)
        logger.info(f"TOOL CALL: {tool_name}")

        if args:
            logger.info(f"  Args: {args}")
        if kwargs:
            logger.info(f"  Kwargs: {kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.info(f"  Result: {result}")
            logger.info("  Status: SUCCESS")
            return result

        except Exception as e:
            logger.error(f"  Error: {str(e)}", exc_info=True)
            logger.info("  Status: FAILED")
            raise
        finally:
            logger.info("-" * 60)

    return wrapper


# ============================================================================
# ENVIRONMENT INITIALIZATION WITH CONFIG
# ============================================================================


def initialize_environment(config: ConfigManager):
    """Initialize the robot environment using configuration."""
    global env, robot

    logger.info("=" * 60)
    logger.info("ENVIRONMENT INITIALIZATION")
    logger.info(f"  Robot Type: {config.robot.type}")
    logger.info(f"  Simulation: {config.robot.simulation}")
    logger.info(f"  Camera: {config.robot.enable_camera}")
    logger.info(f"  Camera Rate: {config.robot.camera_update_rate_hz} Hz")
    logger.info(f"  Verbose: {config.robot.verbose}")
    logger.info("=" * 60)

    # Get ElevenLabs API key from environment
    el_api_key = os.getenv("ELEVENLABS_API_KEY", "")

    env = Environment(
        el_api_key=el_api_key,
        use_simulation=config.robot.simulation,
        robot_id=config.robot.type,
        verbose=config.robot.verbose,
        start_camera_thread=config.robot.enable_camera,
    )
    robot = env.robot()

    logger.info("Environment initialized successfully")
    logger.info("=" * 60)


# ============================================================================
# MCP SETUP
# ============================================================================

mcp = FastMCP("robot-environment")


# ============================================================================
# TOOL DEFINITIONS (same as before, but using config values)
# ============================================================================


@mcp.tool
@log_tool_call
@validate_input(PickPlaceInput)
def pick_place_object(
    object_name: str,
    pick_coordinate: List[float],
    place_coordinate: List[float],
    location: Union[Location, str, None] = None,
    z_offset: float = None,  # Will use config default if None
) -> str:
    """Pick and place an object using configured motion parameters."""

    # Use configured z_offset if not provided
    if z_offset is None:
        z_offset = config.robot.motion.pick_z_offset

    try:
        success = robot.pick_place_object(
            object_name=object_name,
            pick_coordinate=pick_coordinate,
            place_coordinate=place_coordinate,
            location=location,
            z_offset=z_offset,
        )

        if success:
            location_str = f" {location} coordinate" if location else " at"
            z_offset_str = f" (z_offset: {z_offset:.3f}m)" if z_offset != config.robot.motion.pick_z_offset else ""
            return (
                f"✓ Successfully picked '{object_name}' from [{pick_coordinate[0]:.3f}, {pick_coordinate[1]:.3f}]{z_offset_str} "
                f"and placed it{location_str} [{place_coordinate[0]:.3f}, {place_coordinate[1]:.3f}]"
            )
        else:
            return f"❌ Failed to pick and place '{object_name}'"
    except Exception as e:
        return f"❌ Error during pick_place_object: {str(e)}"


# [Include all other tool definitions from original server - abbreviated for space]
# get_detected_objects, pick_object, place_object, push_object, move2observation_pose,
# get_largest_free_space_with_center, etc.


# ============================================================================
# MAIN ENTRY POINT WITH CONFIG MANAGEMENT
# ============================================================================


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FastMCP Robot Server with Configuration Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python server/fastmcp_robot_server_with_config.py

  # Specify custom config file
  python server/fastmcp_robot_server_with_config.py --config my_config.yaml

  # Use production environment
  ROBOT_ENV=production python server/fastmcp_robot_server_with_config.py

  # Override specific settings
  python server/fastmcp_robot_server_with_config.py --robot-type widowx --port 8080

  # Development mode with debug logging
  ROBOT_ENV=development python server/fastmcp_robot_server_with_config.py
        """,
    )

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to configuration YAML file (default: config/robot_config.yaml)")

    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "testing"],
        help="Environment name for config overrides (or set ROBOT_ENV)",
    )

    # Runtime overrides
    parser.add_argument("--host", type=str, help="Override server host")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--robot-type", type=str, choices=["niryo", "widowx"], help="Override robot type")
    parser.add_argument("--simulation", action="store_true", help="Force simulation mode")
    parser.add_argument("--no-simulation", action="store_true", help="Force real robot mode")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def apply_cli_overrides(config: ConfigManager, args):
    """Apply command-line argument overrides to configuration."""

    if args.host:
        config.set("server.host", args.host)
        logger.info(f"CLI Override: server.host = {args.host}")

    if args.port:
        config.set("server.port", args.port)
        logger.info(f"CLI Override: server.port = {args.port}")

    if args.robot_type:
        config.set("robot.type", args.robot_type)
        logger.info(f"CLI Override: robot.type = {args.robot_type}")

    if args.simulation:
        config.set("robot.simulation", True)
        logger.info("CLI Override: robot.simulation = True")

    if args.no_simulation:
        config.set("robot.simulation", False)
        logger.info("CLI Override: robot.simulation = False")

    if args.no_camera:
        config.set("robot.enable_camera", False)
        logger.info("CLI Override: robot.enable_camera = False")

    if args.verbose:
        config.set("robot.verbose", True)
        logger.info("CLI Override: robot.verbose = True")


def print_startup_info(config: ConfigManager):
    """Print startup information to console."""
    print("\n" + "=" * 70)
    print("FASTMCP ROBOT SERVER - Configuration Management")
    print("=" * 70)
    print("\n📋 Configuration Summary:")
    print(f"  • Server:      {config.server.host}:{config.server.port}")
    print(f"  • Robot:       {config.robot.type.upper()}")
    print(f"  • Mode:        {'Simulation' if config.robot.simulation else 'Real Robot'}")
    print(f"  • Camera:      {'Enabled' if config.robot.enable_camera else 'Disabled'}")
    print(f"  • Log Level:   {config.server.log_level}")
    print(f"  • LLM Provider: {config.llm.default_provider.upper()}")

    print("\n🔧 Detection Settings:")
    print(f"  • Model:       {config.detection.model.upper()}")
    print(f"  • Device:      {config.detection.device.upper()}")
    print(f"  • Confidence:  {config.detection.confidence_threshold}")
    print(f"  • Labels:      {len(config.detection.default_labels)} objects")

    print("\n🤖 Motion Parameters:")
    print(f"  • Pick Z-offset:   {config.robot.motion.pick_z_offset:.3f}m")
    print(f"  • Place Z-offset:  {config.robot.motion.place_z_offset:.3f}m")
    print(f"  • Safe height:     {config.robot.motion.safe_height:.3f}m")
    print(f"  • Approach speed:  {config.robot.motion.approach_speed}%")

    workspace_key = config.robot.type
    if workspace_key in config.robot.workspace:
        ws = config.robot.workspace[workspace_key]
        print(f"\n📐 Workspace ({ws.id}):")
        print(f"  • X range: [{ws.bounds.x_min:.3f}, {ws.bounds.x_max:.3f}] m")
        print(f"  • Y range: [{ws.bounds.y_min:.3f}, {ws.bounds.y_max:.3f}] m")
        print(f"  • Center:  [{ws.center[0]:.3f}, {ws.center[1]:.3f}] m")

    print("\n🌐 Endpoints:")
    print(f"  • HTTP:  http://{config.server.host}:{config.server.port}")
    print(f"  • SSE:   http://{config.server.host}:{config.server.port}/sse")

    print(f"\n📝 Logs: {config.server.log_dir}/mcp_server_*.log")
    print("\n✨ Press Ctrl+C to stop")
    print("=" * 70 + "\n")


def main():
    """Main entry point with configuration management."""
    global config, logger

    # Parse arguments
    args = parse_arguments()

    try:
        # Load configuration
        print("📂 Loading configuration...")
        config = load_config(config_path=args.config, environment=args.environment or os.getenv("ROBOT_ENV"))
        print("✓ Configuration loaded successfully\n")

    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print("\nPlease create config/robot_config.yaml or specify --config")
        sys.exit(1)

    except ValidationError as e:
        print("❌ Configuration validation failed:")
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            print(f"  • {loc}: {error['msg']}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(config)

    # Apply CLI overrides
    if any([args.host, args.port, args.robot_type, args.simulation, args.no_simulation, args.no_camera, args.verbose]):
        logger.info("Applying command-line overrides...")
        apply_cli_overrides(config, args)

    # Log full configuration
    logger.info("=" * 80)
    logger.info("LOADED CONFIGURATION")
    logger.info("=" * 80)
    for key, value in config.to_dict().items():
        if key != "environments":  # Skip environment overrides in log
            logger.info(f"{key}: {value}")
    logger.info("=" * 80)

    # Print startup info
    print_startup_info(config)

    # Initialize environment
    try:
        initialize_environment(config)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}", exc_info=True)
        print(f"\n❌ Failed to initialize environment: {e}")
        sys.exit(1)

    # Run server
    logger.info("Starting MCP server...")

    try:
        mcp.run(transport="sse", host=config.server.host, port=config.server.port)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        print("\n\n✓ Server stopped gracefully")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        print(f"\n❌ Server error: {e}")
        raise
    finally:
        logger.info("=" * 80)
        logger.info("SERVER STOPPED")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

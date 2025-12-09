# config/config_manager.py
"""
Configuration Management System for Robot MCP

Centralizes configuration loading, validation, and access across the system.
Supports YAML config files, environment variables, and environment-specific overrides.

Usage:
    from config.config_manager import ConfigManager

    # Load configuration
    config = ConfigManager.load()

    # Access settings
    host = config.server.host
    port = config.server.port
    robot_type = config.robot.type

    # Get nested values
    x_min = config.get("robot.workspace.niryo.bounds.x_min")

    # Override at runtime
    config.set("llm.temperature", 0.9)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

# ============================================================================
# PYDANTIC MODELS FOR TYPE-SAFE CONFIGURATION
# ============================================================================


class ServerConfig(BaseModel):
    """Server configuration settings."""

    host: str = Field("127.0.0.1", description="Server host address")
    port: int = Field(8000, ge=1024, le=65535, description="Server port")
    max_workers: int = Field(4, ge=1, description="Maximum worker threads")
    log_level: str = Field("INFO", description="Logging level")
    log_dir: str = Field("log", description="Log directory path")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class WorkspaceBounds(BaseModel):
    """Workspace boundary coordinates."""

    x_min: float = Field(..., description="Minimum X coordinate (meters)")
    x_max: float = Field(..., description="Maximum X coordinate (meters)")
    y_min: float = Field(..., description="Minimum Y coordinate (meters)")
    y_max: float = Field(..., description="Maximum Y coordinate (meters)")

    @field_validator("x_max")
    @classmethod
    def validate_x_range(cls, v: float, info) -> float:
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def validate_y_range(cls, v: float, info) -> float:
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return v


class WorkspaceConfig(BaseModel):
    """Workspace configuration."""

    id: str = Field(..., description="Workspace identifier")
    bounds: WorkspaceBounds = Field(..., description="Workspace boundaries")
    center: list[float] = Field(..., description="Workspace center [x, y]")

    @field_validator("center")
    @classmethod
    def validate_center(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("center must be [x, y] coordinates")
        return v


class MotionConfig(BaseModel):
    """Robot motion parameters."""

    pick_z_offset: float = Field(0.001, ge=0.0, le=0.1, description="Z-offset for picking (m)")
    place_z_offset: float = Field(0.001, ge=0.0, le=0.1, description="Z-offset for placing (m)")
    safe_height: float = Field(0.15, ge=0.05, le=0.5, description="Safe height (m)")
    approach_speed: int = Field(50, ge=1, le=100, description="Approach speed (%)")
    retract_speed: int = Field(50, ge=1, le=100, description="Retract speed (%)")
    gripper_close_delay: float = Field(0.5, ge=0.0, le=5.0, description="Gripper close delay (s)")
    gripper_open_delay: float = Field(0.3, ge=0.0, le=5.0, description="Gripper open delay (s)")


class RobotConfig(BaseModel):
    """Robot configuration settings."""

    type: str = Field("niryo", description="Robot type")
    simulation: bool = Field(True, description="Use simulation mode")
    verbose: bool = Field(False, description="Enable verbose output")
    enable_camera: bool = Field(True, description="Enable camera")
    camera_update_rate_hz: float = Field(2.0, ge=0.1, le=30.0, description="Camera update rate (Hz)")
    workspace: Dict[str, WorkspaceConfig] = Field(..., description="Workspace configurations")
    motion: MotionConfig = Field(..., description="Motion parameters")

    @field_validator("type")
    @classmethod
    def validate_robot_type(cls, v: str) -> str:
        valid_types = ["niryo", "widowx"]
        if v.lower() not in valid_types:
            raise ValueError(f"robot type must be one of {valid_types}")
        return v.lower()


class SpatialConfig(BaseModel):
    """Spatial query thresholds."""

    close_to_radius_m: float = Field(0.02, ge=0.001, le=0.5, description="'Close to' radius (m)")
    left_right_threshold_m: float = Field(0.01, ge=0.001, le=0.1, description="Left/right threshold (m)")
    above_below_threshold_m: float = Field(0.01, ge=0.001, le=0.1, description="Above/below threshold (m)")


class DetectionConfig(BaseModel):
    """Object detection settings."""

    model: str = Field("owlv2", description="Detection model")
    device: str = Field("cuda", description="Computation device")
    confidence_threshold: float = Field(0.15, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0, description="IoU threshold")
    max_detections: int = Field(100, ge=1, le=1000, description="Maximum detections")
    default_labels: list[str] = Field(default_factory=list, description="Default object labels")
    spatial: SpatialConfig = Field(..., description="Spatial query thresholds")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid_models = ["owlv2", "yoloworld"]
        if v.lower() not in valid_models:
            raise ValueError(f"model must be one of {valid_models}")
        return v.lower()

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices = ["cuda", "cpu"]
        if v.lower() not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        return v.lower()


class LLMProviderConfig(BaseModel):
    """LLM provider-specific settings."""

    enabled: bool = Field(True, description="Enable this provider")
    default_model: str = Field(..., description="Default model name")
    models: list[str] = Field(default_factory=list, description="Available models")
    rate_limit_rpm: Optional[int] = Field(None, ge=1, description="Rate limit (requests/min)")
    base_url: Optional[str] = Field(None, description="Base URL for API")


class LLMConfig(BaseModel):
    """LLM configuration settings."""

    default_provider: str = Field("auto", description="Default LLM provider")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(4096, ge=1, le=128000, description="Maximum tokens")
    enable_cot: bool = Field(True, description="Enable chain-of-thought")
    require_planning: bool = Field(True, description="Require planning phase")
    max_iterations: int = Field(15, ge=1, le=50, description="Max tool-calling iterations")
    providers: Dict[str, LLMProviderConfig] = Field(..., description="Provider configurations")

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["auto", "openai", "groq", "gemini", "ollama"]
        if v.lower() not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}")
        return v.lower()


class TTSProviderConfig(BaseModel):
    """TTS provider settings."""

    voice_id: Optional[str] = None
    voice: Optional[str] = None
    stability: Optional[float] = Field(None, ge=0.0, le=1.0)
    similarity_boost: Optional[float] = Field(None, ge=0.0, le=1.0)
    speed: Optional[float] = Field(None, ge=0.1, le=2.0)


class TTSConfig(BaseModel):
    """Text-to-speech settings."""

    enabled: bool = Field(True, description="Enable TTS")
    provider: str = Field("elevenlabs", description="TTS provider")
    elevenlabs: TTSProviderConfig = Field(default_factory=TTSProviderConfig)
    kokoro: TTSProviderConfig = Field(default_factory=TTSProviderConfig)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["elevenlabs", "kokoro", "none"]
        if v.lower() not in valid_providers:
            raise ValueError(f"TTS provider must be one of {valid_providers}")
        return v.lower()


class RedisStreamsConfig(BaseModel):
    """Redis stream names."""

    camera: str = Field("robot_camera", description="Camera stream name")
    detected_objects: str = Field("detected_objects", description="Detected objects stream")
    annotated_frames: str = Field("annotated_frames", description="Annotated frames stream")


class RedisConfig(BaseModel):
    """Redis connection settings."""

    host: str = Field("localhost", description="Redis host")
    port: int = Field(6379, ge=1, le=65535, description="Redis port")
    db: int = Field(0, ge=0, le=15, description="Redis database number")
    decode_responses: bool = Field(True, description="Decode responses")
    streams: RedisStreamsConfig = Field(..., description="Stream names")


class GUIConfig(BaseModel):
    """GUI settings."""

    host: str = Field("127.0.0.1", description="GUI host")
    port: int = Field(7860, ge=1024, le=65535, description="GUI port")
    share: bool = Field(False, description="Create public link")
    enable_voice_input: bool = Field(True, description="Enable voice input")
    enable_camera_feed: bool = Field(True, description="Enable camera feed")
    theme: str = Field("default", description="GUI theme")


class LogRotationConfig(BaseModel):
    """Log rotation settings."""

    max_bytes: int = Field(10485760, ge=1024, description="Max log file size (bytes)")
    backup_count: int = Field(5, ge=1, le=100, description="Number of backup files")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    format: str = Field(..., description="Log message format")
    date_format: str = Field("%Y-%m-%d %H:%M:%S", description="Date format")
    levels: Dict[str, str] = Field(..., description="Log levels by module")
    rotation: LogRotationConfig = Field(..., description="Log rotation settings")


class EnvironmentOverrides(BaseModel):
    """Environment-specific configuration overrides."""

    server: Optional[Dict[str, Any]] = None
    robot: Optional[Dict[str, Any]] = None
    llm: Optional[Dict[str, Any]] = None


class RobotMCPConfig(BaseModel):
    """Root configuration model."""

    server: ServerConfig
    robot: RobotConfig
    detection: DetectionConfig
    llm: LLMConfig
    tts: TTSConfig
    redis: RedisConfig
    gui: GUIConfig
    logging: LoggingConfig
    environments: Optional[Dict[str, EnvironmentOverrides]] = None


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================


class ConfigManager:
    """
    Configuration manager with validation and environment support.

    Features:
    - Load from YAML files with validation
    - Environment-specific overrides
    - Environment variable substitution
    - Runtime configuration updates
    - Dot-notation access to nested values

    Example:
        >>> config = ConfigManager.load()
        >>> print(config.server.host)
        127.0.0.1
        >>> config.set("llm.temperature", 0.9)
        >>> temp = config.get("llm.temperature")
        0.9
    """

    _instance: Optional["ConfigManager"] = None
    _config: Optional[RobotMCPConfig] = None

    def __init__(self, config: RobotMCPConfig):
        """Initialize with validated config."""
        self._config = config

    @classmethod
    def load(cls, config_path: Optional[str] = None, environment: Optional[str] = None) -> "ConfigManager":
        """
        Load configuration from YAML file with validation.

        Args:
            config_path: Path to config file (default: config/robot_config.yaml)
            environment: Environment name for overrides (dev, prod, test)
                        Can also be set via ROBOT_ENV environment variable

        Returns:
            ConfigManager instance with validated configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
        """
        # Determine config file path
        if config_path is None:
            # Look for config in standard locations
            possible_paths = ["config/robot_config.yaml", "robot_config.yaml", "../config/robot_config.yaml"]

            for path in possible_paths:
                if Path(path).exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find robot_config.yaml. " "Please create config/robot_config.yaml or specify config_path"
                )

        # Load YAML
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Substitute environment variables
        raw_config = cls._substitute_env_vars(raw_config)

        # Apply environment-specific overrides
        environment = environment or os.getenv("ROBOT_ENV")
        if environment and "environments" in raw_config:
            if environment in raw_config["environments"]:
                overrides = raw_config["environments"][environment]
                raw_config = cls._merge_dicts(raw_config, overrides)

        # Validate with Pydantic
        try:
            validated_config = RobotMCPConfig(**raw_config)
        except ValidationError as e:
            print("❌ Configuration validation failed:")
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                print(f"  • {loc}: {error['msg']}")
            raise

        # Create singleton instance
        instance = cls(validated_config)
        cls._instance = instance

        return instance

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """Get singleton instance."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized. Call ConfigManager.load() first.")
        return cls._instance

    @staticmethod
    def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively substitute environment variables in config.

        Supports ${VAR_NAME} and ${VAR_NAME:default} syntax.
        """
        if isinstance(config, dict):
            return {k: ConfigManager._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigManager._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Check for ${VAR} or ${VAR:default} pattern
            import re

            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

            def replace(match):
                var_name = match.group(1)
                default_value = match.group(2)
                return os.getenv(var_name, default_value or match.group(0))

            return re.sub(pattern, replace, config)
        else:
            return config

    @staticmethod
    def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override dict into base dict."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            path: Dot-separated path (e.g., "server.host")
            default: Default value if path not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("robot.workspace.niryo.bounds.x_min")
            0.163
        """
        keys = path.split(".")
        value = self._config

        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    value = getattr(value, key)
            return value
        except (KeyError, AttributeError):
            return default

    def set(self, path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            path: Dot-separated path (e.g., "llm.temperature")
            value: New value

        Example:
            >>> config.set("llm.temperature", 0.9)
        """
        keys = path.split(".")
        obj = self._config

        # Navigate to parent
        for key in keys[:-1]:
            if isinstance(obj, dict):
                obj = obj[key]
            else:
                obj = getattr(obj, key)

        # Set final value
        final_key = keys[-1]
        if isinstance(obj, dict):
            obj[final_key] = value
        else:
            setattr(obj, final_key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config.model_dump()

    def save(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to output file
        """
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # Convenience properties for direct access
    @property
    def server(self) -> ServerConfig:
        return self._config.server

    @property
    def robot(self) -> RobotConfig:
        return self._config.robot

    @property
    def detection(self) -> DetectionConfig:
        return self._config.detection

    @property
    def llm(self) -> LLMConfig:
        return self._config.llm

    @property
    def tts(self) -> TTSConfig:
        return self._config.tts

    @property
    def redis(self) -> RedisConfig:
        return self._config.redis

    @property
    def gui(self) -> GUIConfig:
        return self._config.gui

    @property
    def logging(self) -> LoggingConfig:
        return self._config.logging

    def __repr__(self) -> str:
        return f"ConfigManager(server={self.server.host}:{self.server.port}, robot={self.robot.type})"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def load_config(config_path: Optional[str] = None, environment: Optional[str] = None) -> ConfigManager:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to config file
        environment: Environment name (dev, prod, test)

    Returns:
        ConfigManager instance
    """
    return ConfigManager.load(config_path, environment)


def get_config() -> ConfigManager:
    """Get singleton ConfigManager instance."""
    return ConfigManager.get_instance()

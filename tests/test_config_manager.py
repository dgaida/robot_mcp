"""Unit tests for configuration management."""

import os

import pytest
import yaml

from config.config_manager import ConfigManager


def test_config_load_default(tmp_path):
    """Test loading default configuration."""
    # Create a minimal valid config
    config_data = {
        "server": {"host": "127.0.0.1", "port": 8000},
        "robot": {
            "type": "niryo",
            "workspace": {
                "niryo": {
                    "id": "niryo_ws",
                    "bounds": {"x_min": 0.1, "x_max": 0.2, "y_min": 0.1, "y_max": 0.2},
                    "center": [0.15, 0.15],
                }
            },
            "motion": {},
        },
        "detection": {"spatial": {}},
        "llm": {"providers": {"openai": {"default_model": "gpt-4o"}}},
        "tts": {},
        "redis": {"streams": {}},
        "gui": {},
        "logging": {"format": "%(message)s", "levels": {}, "rotation": {}},
    }

    config_file = tmp_path / "robot_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = ConfigManager.load(config_path=str(config_file))
    assert config.server.host == "127.0.0.1"
    assert config.robot.type == "niryo"
    assert "niryo" in config.robot.workspace


def test_config_with_environment_override(tmp_path):
    """Test environment-specific overrides."""
    config_data = {
        "server": {"host": "127.0.0.1", "port": 8000},
        "robot": {
            "type": "niryo",
            "workspace": {
                "niryo": {
                    "id": "niryo_ws",
                    "bounds": {"x_min": 0.1, "x_max": 0.2, "y_min": 0.1, "y_max": 0.2},
                    "center": [0.15, 0.15],
                }
            },
            "motion": {},
        },
        "detection": {"spatial": {}},
        "llm": {"providers": {"openai": {"default_model": "gpt-4o"}}},
        "tts": {},
        "redis": {"streams": {}},
        "gui": {},
        "logging": {"format": "%(message)s", "levels": {}, "rotation": {}},
        "environments": {"development": {"server": {"log_level": "DEBUG"}, "robot": {"simulation": True}}},
    }

    config_file = tmp_path / "robot_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Test partial override in development environment
    os.environ["ROBOT_ENV"] = "development"
    config = ConfigManager.load(config_path=str(config_file))

    assert config.server.log_level == "DEBUG"
    assert config.robot.simulation is True
    # Ensure non-overridden fields are still there
    assert config.server.host == "127.0.0.1"
    assert "niryo" in config.robot.workspace


def test_config_invalid_missing_workspace(tmp_path):
    """Test missing workspace validation."""
    config_data = {
        "server": {"host": "127.0.0.1"},
        "robot": {"type": "niryo", "workspace": {}},  # Empty workspace
        "detection": {"spatial": {}},
        "llm": {"providers": {"openai": {"default_model": "gpt-4o"}}},
        "tts": {},
        "redis": {"streams": {}},
        "gui": {},
        "logging": {"format": "%(message)s", "levels": {}, "rotation": {}},
    }

    config_file = tmp_path / "robot_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(Exception) as excinfo:
        ConfigManager.load(config_path=str(config_file))
    assert "No workspaces defined" in str(excinfo.value)


def test_config_invalid_zero_area_workspace(tmp_path):
    """Test zero-area workspace validation."""
    config_data = {
        "server": {"host": "127.0.0.1"},
        "robot": {
            "type": "niryo",
            "workspace": {
                "niryo": {
                    "id": "niryo_ws",
                    "bounds": {"x_min": 0.1, "x_max": 0.1, "y_min": 0.1, "y_max": 0.1},  # Zero area
                    "center": [0.1, 0.1],
                }
            },
        },
        "detection": {"spatial": {}},
        "llm": {"providers": {"openai": {"default_model": "gpt-4o"}}},
        "tts": {},
        "redis": {"streams": {}},
        "gui": {},
        "logging": {"format": "%(message)s", "levels": {}, "rotation": {}},
    }

    config_file = tmp_path / "robot_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(Exception) as excinfo:
        ConfigManager.load(config_path=str(config_file))
    # It might be caught by field_validator or model_validator
    assert any(msg in str(excinfo.value) for msg in ["zero-area bounds", "must be greater than"])

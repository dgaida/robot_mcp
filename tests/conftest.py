# tests/conftest.py
"""
Pytest configuration and shared fixtures
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# ============================================================================
# WORKAROUNDS FOR BROKEN EXTERNAL DEPENDENCIES
# ============================================================================

# 1. Fix text2speech broken structure (missing engines module)
# This MUST be done before anything imports text2speech or robot_environment
engines_mock = MagicMock()
engines_mock.TTSEngine = MagicMock()
engines_mock.KokoroEngine = MagicMock()
engines_mock.ElevenLabsEngine = MagicMock()
sys.modules["text2speech.engines"] = engines_mock

try:
    import text2speech  # noqa: F401
except ImportError:
    # If text2speech is missing, mock it entirely
    t2s_mock = MagicMock()
    sys.modules["text2speech"] = t2s_mock

# 2. Fix robot_workspace missing Workspaces export (breaks robot_environment)
try:
    # Try to get the actual Workspaces class
    try:
        from robot_workspace.workspaces.workspaces import Workspaces
    except ImportError:
        Workspaces = MagicMock()

    # Ensure robot_workspace module exists in sys.modules
    import robot_workspace

    # Inject Workspaces into the module
    robot_workspace.Workspaces = Workspaces
    sys.modules["robot_workspace"].Workspaces = Workspaces
except ImportError:
    # If robot_workspace is missing, mock it
    rw_mock = MagicMock()
    rw_mock.Workspaces = MagicMock()
    rw_mock.Location = MagicMock()
    sys.modules["robot_workspace"] = rw_mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_coordinates():
    """Sample coordinate sets for testing."""
    return {
        "center": [0.25, 0.0],
        "upper_left": [0.337, 0.087],
        "upper_right": [0.337, -0.087],
        "lower_left": [0.163, 0.087],
        "lower_right": [0.163, -0.087],
    }


@pytest.fixture
def sample_objects():
    """Sample object data for testing."""
    return [
        {
            "label": "pencil",
            "x": 0.15,
            "y": -0.05,
            "width_m": 0.015,
            "height_m": 0.120,
            "area_m2": 0.0018,
            "rotation_rad": 0.785,
        },
        {"label": "cube", "x": 0.20, "y": 0.10, "width_m": 0.050, "height_m": 0.050, "area_m2": 0.0025, "rotation_rad": 0.0},
        {
            "label": "cylinder",
            "x": 0.28,
            "y": 0.0,
            "width_m": 0.040,
            "height_m": 0.080,
            "area_m2": 0.0032,
            "rotation_rad": 1.571,
        },
    ]


@pytest.fixture
def mock_groq_api_key():
    """Mock Groq API key for testing."""
    return "test_groq_api_key_1234567890"


@pytest.fixture
def mock_elevenlabs_api_key():
    """Mock ElevenLabs API key for testing."""
    return "test_elevenlabs_api_key_1234567890"


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker if no other markers present
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

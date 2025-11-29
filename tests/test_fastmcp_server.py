# tests/test_fastmcp_server.py
"""
Unit tests for FastMCP Robot Server

Note: Server functions are decorated with @mcp.tool, which wraps them.
To test, we need to access the underlying function via .func attribute.
"""

from unittest.mock import MagicMock, patch

import pytest
from robot_environment.robot.robot_api import Location

import server.fastmcp_robot_server as server_module


class TestEnvironmentInitialization:
    """Test environment initialization."""

    @patch("server.fastmcp_robot_server.Environment")
    def test_initialize_environment(self, mock_env_class):
        """Test environment initialization with default parameters."""
        mock_env = MagicMock()
        mock_robot = MagicMock()
        mock_env.robot.return_value = mock_robot
        mock_env_class.return_value = mock_env

        server_module.initialize_environment(
            el_api_key="test_key", use_simulation=True, robot_id="niryo", verbose=True, start_camera_thread=True
        )

        mock_env_class.assert_called_once_with(
            el_api_key="test_key", use_simulation=True, robot_id="niryo", verbose=True, start_camera_thread=True
        )

    @patch("server.fastmcp_robot_server.Environment")
    def test_initialize_with_simulation_disabled(self, mock_env_class):
        """Test initialization with real robot."""
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env

        server_module.initialize_environment(use_simulation=False, robot_id="widowx")

        args, kwargs = mock_env_class.call_args
        assert kwargs["use_simulation"] is False
        assert kwargs["robot_id"] == "widowx"


class TestRobotTools:
    """Test robot control tools."""

    def setup_method(self):
        """Setup test fixtures."""
        with patch("server.fastmcp_robot_server.Environment"):
            server_module.initialize_environment()

        self.mock_robot = MagicMock()
        server_module.robot = self.mock_robot

    def test_pick_place_object(self):
        """Test pick and place object."""
        self.mock_robot.pick_place_object.return_value = True

        # Access the underlying function via .func attribute
        pick_place_func = server_module.pick_place_object.func
        result = pick_place_func(
            object_name="pencil", pick_coordinate=[0.15, -0.05], place_coordinate=[0.20, 0.10], location="right next to"
        )

        assert "Successfully picked" in result
        self.mock_robot.pick_place_object.assert_called_once_with(
            object_name="pencil", pick_coordinate=[0.15, -0.05], place_coordinate=[0.20, 0.10], location="right next to"
        )

    def test_pick_place_object_with_location_enum(self):
        """Test pick and place with Location enum."""
        self.mock_robot.pick_place_object.return_value = True

        pick_place_func = server_module.pick_place_object.func
        result = pick_place_func(
            object_name="cube", pick_coordinate=[0.2, 0.0], place_coordinate=[0.25, 0.05], location=Location.LEFT_NEXT_TO
        )

        assert "Successfully picked" in result
        assert self.mock_robot.pick_place_object.called

    def test_pick_object(self):
        """Test pick object."""
        self.mock_robot.pick_object.return_value = True

        pick_func = server_module.pick_object.func
        result = pick_func(object_name="pen", pick_coordinate=[0.18, -0.03])

        assert "Successfully picked" in result
        self.mock_robot.pick_object.assert_called_once()

    def test_place_object(self):
        """Test place object."""
        self.mock_robot.place_object.return_value = True

        place_func = server_module.place_object.func
        result = place_func(place_coordinate=[0.20, 0.05], location="left next to")

        assert "Successfully placed" in result
        self.mock_robot.place_object.assert_called_once()

    def test_push_object(self):
        """Test push object."""
        self.mock_robot.push_object.return_value = True

        push_func = server_module.push_object.func
        result = push_func(object_name="large_box", push_coordinate=[0.25, 0.05], direction="right", distance=50.0)

        assert "Successfully pushed" in result
        self.mock_robot.push_object.assert_called_once_with("large_box", [0.25, 0.05], "right", 50.0)


class TestObjectDetectionTools:
    """Test object detection tools."""

    def setup_method(self):
        """Setup test fixtures."""
        with patch("server.fastmcp_robot_server.Environment") as mock_env_class:
            self.mock_env = MagicMock()
            mock_env_class.return_value = self.mock_env
            server_module.initialize_environment()

        server_module.env = self.mock_env

        # Create mock objects
        self.mock_object1 = MagicMock()
        self.mock_object1.label.return_value = "pencil"
        self.mock_object1.x_com.return_value = 0.15
        self.mock_object1.y_com.return_value = -0.05
        self.mock_object1.width_m.return_value = 0.015
        self.mock_object1.height_m.return_value = 0.120
        self.mock_object1.size_m2.return_value = 0.0018
        self.mock_object1.gripper_rotation.return_value = 0.785

        self.mock_object2 = MagicMock()
        self.mock_object2.label.return_value = "cube"
        self.mock_object2.x_com.return_value = 0.20
        self.mock_object2.y_com.return_value = 0.10
        self.mock_object2.width_m.return_value = 0.050
        self.mock_object2.height_m.return_value = 0.050
        self.mock_object2.size_m2.return_value = 0.0025
        self.mock_object2.gripper_rotation.return_value = 0.0

    def test_get_detected_objects_all(self):
        """Test getting all detected objects."""
        mock_objects = MagicMock()
        mock_objects.get_detected_objects_serializable.return_value = [
            {"label": "pencil", "x": 0.15, "y": -0.05},
            {"label": "cube", "x": 0.20, "y": 0.10},
        ]
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_objects_func = server_module.get_detected_objects.func
        result = get_objects_func()

        assert "Found 2 object(s)" in result
        assert "pencil" in result

    def test_get_detected_objects_with_label_filter(self):
        """Test getting objects with label filter."""
        mock_objects = MagicMock()
        mock_objects.get_detected_objects_serializable.return_value = [{"label": "pencil", "x": 0.15, "y": -0.05}]
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_objects_func = server_module.get_detected_objects.func
        result = get_objects_func(label="pencil")

        assert "Found 1 object(s)" in result
        assert "pencil" in result

    def test_get_detected_objects_with_location_filter(self):
        """Test getting objects with location filter."""
        mock_objects = MagicMock()
        mock_objects.get_detected_objects_serializable.return_value = [{"label": "cube", "x": 0.20, "y": 0.10}]
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_objects_func = server_module.get_detected_objects.func
        result = get_objects_func(location="left next to", coordinate=[0.15, 0.0])

        assert "Found 1 object(s)" in result
        assert "cube" in result

    def test_get_detected_object(self):
        """Test getting specific object at coordinate."""
        mock_objects = MagicMock()
        mock_objects.get_detected_object.return_value = {"label": "pencil", "x": 0.15, "y": -0.05}
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_object_func = server_module.get_detected_object.func
        result = get_object_func([0.15, -0.05])

        assert "Found object" in result
        assert "pencil" in result

    def test_get_detected_object_not_found(self):
        """Test when object not found at coordinate."""
        mock_objects = MagicMock()
        mock_objects.get_detected_object.return_value = None
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_object_func = server_module.get_detected_object.func
        result = get_object_func([0.99, 0.99])

        assert "No object found" in result

    def test_get_largest_detected_object(self):
        """Test getting largest object."""
        mock_objects = MagicMock()
        mock_objects.get_largest_detected_object.return_value = ({"label": "cube", "x": 0.20, "y": 0.10}, 0.0025)
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_largest_func = server_module.get_largest_detected_object.func
        result = get_largest_func()

        assert "Largest object" in result
        assert "cube" in result

    def test_get_smallest_detected_object(self):
        """Test getting smallest object."""
        mock_objects = MagicMock()
        mock_objects.get_smallest_detected_object.return_value = ({"label": "pencil", "x": 0.15, "y": -0.05}, 0.0018)
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_smallest_func = server_module.get_smallest_detected_object.func
        result = get_smallest_func()

        assert "Smallest object" in result
        assert "pencil" in result

    def test_get_detected_objects_sorted_ascending(self):
        """Test getting sorted objects (ascending)."""
        mock_objects = MagicMock()
        mock_objects.get_detected_objects_sorted.return_value = [
            {"label": "pencil", "area": 0.0018},
            {"label": "cube", "area": 0.0025},
        ]
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_sorted_func = server_module.get_detected_objects_sorted.func
        result = get_sorted_func(ascending=True)

        assert "smallest to largest" in result
        assert "pencil" in result

    def test_get_detected_objects_sorted_descending(self):
        """Test getting sorted objects (ascending)."""
        mock_objects = MagicMock()
        mock_objects.get_detected_objects_sorted.return_value = [
            {"label": "pencil", "area": 0.0018},
            {"label": "cube", "area": 0.0025},
        ]
        self.mock_env.get_detected_objects.return_value = mock_objects

        get_sorted_func = server_module.get_detected_objects_sorted.func
        result = get_sorted_func(ascending=False)

        assert "largest to smallest" in result
        assert "cube" in result


class TestWorkspaceTools:
    """Test workspace-related tools."""

    def setup_method(self):
        """Setup test fixtures."""
        with patch("server.fastmcp_robot_server.Environment") as mock_env_class:
            self.mock_env = MagicMock()
            mock_env_class.return_value = self.mock_env
            server_module.initialize_environment()

        server_module.env = self.mock_env

    def test_get_largest_free_space(self):
        """Test getting largest free space."""
        self.mock_env.get_largest_free_space_with_center.return_value = (0.0144, 0.25, 0.0)

        get_free_space_func = server_module.get_largest_free_space_with_center.func
        result = get_free_space_func()

        assert "Largest free space" in result
        assert "0.0144" in result

    def test_get_workspace_coordinate_upper_left(self):
        """Test getting upper left corner coordinate."""
        self.mock_env.get_workspace_coordinate_from_point.return_value = [0.337, 0.087]

        get_coord_func = server_module.get_workspace_coordinate_from_point.func
        result = get_coord_func("niryo_ws", "upper left corner")

        assert "0.337" in result
        assert "0.087" in result

    def test_get_workspace_coordinate_center(self):
        """Test getting center coordinate."""
        self.mock_env.get_workspace_coordinate_from_point.return_value = [0.25, 0.0]

        get_coord_func = server_module.get_workspace_coordinate_from_point.func
        result = get_coord_func("niryo_ws", "center point")

        assert "0.25" in result
        assert "0.0" in result

    def test_get_object_labels(self):
        """Test getting object labels."""
        self.mock_env.get_object_labels_as_string.return_value = "pencil, pen, cube, cylinder, chocolate bar"

        get_labels_func = server_module.get_object_labels_as_string.func
        result = get_labels_func()

        assert "pencil" in result
        assert "cube" in result

    def test_add_object_label(self):
        """Test adding new object label."""
        self.mock_env.add_object_name2object_labels.return_value = "Added 'screwdriver' to recognizable objects"

        add_label_func = server_module.add_object_name2object_labels.func
        result = add_label_func("screwdriver")

        assert "screwdriver" in result


class TestFeedbackTools:
    """Test feedback tools."""

    def setup_method(self):
        """Setup test fixtures."""
        with patch("server.fastmcp_robot_server.Environment") as mock_env_class:
            self.mock_env = MagicMock()
            mock_env_class.return_value = self.mock_env
            server_module.initialize_environment()

        server_module.env = self.mock_env

    def test_speak(self):
        """Test text-to-speech."""
        self.mock_env.oralcom_call_text2speech_async.return_value = None

        speak_func = server_module.speak.func
        result = speak_func("Hello, I am picking up the pencil")

        assert "Speaking:" in result
        assert "Hello" in result
        self.mock_env.oralcom_call_text2speech_async.assert_called_once()


class TestCoordinateValidation:
    """Test coordinate validation and edge cases."""

    def test_valid_coordinates(self):
        """Test with valid coordinates."""
        coords = [0.25, 0.0]
        assert len(coords) == 2
        assert all(isinstance(c, (int, float)) for c in coords)

    def test_workspace_boundaries(self):
        """Test coordinates at workspace boundaries."""
        # Niryo workspace boundaries
        upper_left = [0.337, 0.087]
        lower_right = [0.163, -0.087]

        assert upper_left[0] > lower_right[0]  # X increases forward
        assert upper_left[1] > lower_right[1]  # Y increases left


class TestLocationEnum:
    """Test Location enum handling."""

    def test_location_enum_values(self):
        """Test Location enum values."""
        assert Location.LEFT_NEXT_TO.value == "left next to"
        assert Location.RIGHT_NEXT_TO.value == "right next to"
        assert Location.ABOVE.value == "above"
        assert Location.BELOW.value == "below"
        assert Location.ON_TOP_OF.value == "on top of"
        assert Location.NONE.value == "none"

    def test_location_from_string(self):
        """Test converting string to Location."""
        location_str = "left next to"
        location = Location(location_str)
        assert location == Location.LEFT_NEXT_TO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# schemas.py

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator
from robot_workspace import Location

# ============================================================================
# PYDANTIC MODELS FOR INPUT VALIDATION
# ============================================================================


class CoordinateModel(BaseModel):
    """Validates 2D coordinate [x, y] in meters."""

    coordinate: List[float] = Field(..., min_length=2, max_length=2)

    @field_validator("coordinate")
    @classmethod
    def validate_coordinate_values(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numeric values")
        return v


class PickPlaceInput(BaseModel):
    """Input validation for pick_place_object."""

    object_name: str = Field(..., min_length=1, description="Name of the object to pick")
    pick_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    place_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    location: Optional[Union[Location, str]] = Field(None, description="Relative placement location")
    z_offset: float = Field(0.001, ge=0.0, le=0.1, description="Height offset in meters (0.0-0.1)")

    class Config:
        arbitrary_types_allowed = True  # Allow enum types

    @field_validator("pick_coordinate", "place_coordinate")
    @classmethod
    def validate_coordinates(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numeric values [x, y]")
        return v

    @field_validator("location")
    @classmethod
    def validate_location(cls, v):
        if v is None:
            return v

        # If already a Location enum, validate and return (but check for CLOSE_TO)
        if isinstance(v, Location):
            if v == Location.CLOSE_TO:
                raise ValueError(
                    "Location 'close to' is not valid for pick_place_object. Use specific locations like 'left next to', 'right next to', 'above', 'below', 'on top of', or 'inside'."
                )
            return v

        # If string, validate against Location enum values (excluding 'close to')
        if isinstance(v, str):
            # Get all valid locations except CLOSE_TO and NONE
            valid_locations = [loc.value for loc in Location if loc.value is not None and loc != Location.CLOSE_TO]

            if v.lower() not in [loc.lower() for loc in valid_locations]:
                raise ValueError(
                    f"Invalid location '{v}'. Must be one of: {', '.join(valid_locations)}. Note: 'close to' is not allowed for pick_place_object."
                )
            return v

        raise ValueError(f"Location must be a string or Location enum, got {type(v)}")


class PickObjectInput(BaseModel):
    """Input validation for pick_object."""

    object_name: str = Field(..., min_length=1)
    pick_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    z_offset: float = Field(0.001, ge=0.0, le=0.1, description="Height offset in meters (0.0-0.1)")

    @field_validator("pick_coordinate")
    @classmethod
    def validate_coordinate(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numeric values [x, y]")
        return v


class PlaceObjectInput(BaseModel):
    """Input validation for place_object."""

    place_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    location: Optional[Union[Location, str]] = Field(None, description="Relative placement location")

    @field_validator("place_coordinate")
    @classmethod
    def validate_coordinate(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numeric values [x, y]")
        return v

    @field_validator("location")
    @classmethod
    def validate_location(cls, v):
        if v is None:
            return v

        # If already a Location enum, validate and return
        if isinstance(v, Location):
            return v

        # If string, validate against Location enum values
        if isinstance(v, str):
            valid_locations = [loc.value for loc in Location if loc.value is not None]
            if v.lower() not in [loc.lower() for loc in valid_locations]:
                raise ValueError(f"Invalid location '{v}'. Must be one of: {', '.join(valid_locations)}")
            return v

        raise ValueError(f"Location must be a string or Location enum, got {type(v)}")


class PushObjectInput(BaseModel):
    """Input validation for push_object."""

    object_name: str = Field(..., min_length=1)
    push_coordinate: List[float] = Field(..., min_length=2, max_length=2)
    direction: str = Field(...)
    distance: float = Field(..., gt=0)

    @field_validator("push_coordinate")
    @classmethod
    def validate_coordinate(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numeric values [x, y]")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        valid_directions = ["up", "down", "left", "right"]
        if v.lower() not in valid_directions:
            raise ValueError(f"Direction must be one of: {', '.join(valid_directions)}")
        return v


class WorkspacePointInput(BaseModel):
    """Input validation for get_workspace_coordinate_from_point."""

    workspace_id: str = Field(..., min_length=1)
    point: str = Field(...)

    @field_validator("point")
    @classmethod
    def validate_point(cls, v):
        valid_points = ["upper left corner", "upper right corner", "lower left corner", "lower right corner", "center point"]
        if v.lower() not in valid_points:
            raise ValueError(f"Point must be one of: {', '.join(valid_points)}")
        return v


class GetDetectedObjectsInput(BaseModel):
    """Input validation for get_detected_objects."""

    location: Optional[Union[Location, str]] = Field(None, description="Relative location")
    coordinate: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    label: Optional[str] = None

    @field_validator("coordinate")
    @classmethod
    def validate_coordinate(cls, v):
        if v is not None and not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numeric values [x, y]")
        return v

    @field_validator("location")
    @classmethod
    def validate_location(cls, v):
        if v is None:
            return v

        # If already a Location enum, validate and return
        if isinstance(v, Location):
            return v

        # If string, validate against Location enum values
        if isinstance(v, str):
            valid_locations = [loc.value for loc in Location if loc.value is not None]
            if v.lower() not in [loc.lower() for loc in valid_locations]:
                raise ValueError(f"Invalid location '{v}'. Must be one of: {', '.join(valid_locations)}")
            return v

        raise ValueError(f"Location must be a string or Location enum, got {type(v)}")

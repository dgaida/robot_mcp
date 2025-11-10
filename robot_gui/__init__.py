"""
Robot GUI Package

Gradio-based GUI for robot control with MCP integration.
"""

from .mcp_app import RobotMCPGUI, create_gradio_interface

__all__ = ["RobotMCPGUI", "create_gradio_interface"]

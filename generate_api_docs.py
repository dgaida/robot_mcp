#!/usr/bin/env python3
"""
Generate API documentation from FastMCP tool definitions.

This script extracts tool information from fastmcp_robot_server.py
(or any unified server) and generates comprehensive markdown documentation.

Features:
- Extracts @mcp.tool decorated functions
- Parses docstrings (Google/NumPy/reStructuredText style)
- Extracts type hints and Pydantic validation schemas
- Generates examples from docstrings
- Creates categorized API reference
- Validates documentation completeness
"""

import ast

# import inspect
# import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))


@dataclass
class Parameter:
    """Represents a function parameter."""

    name: str
    type_hint: str
    default: Optional[str]
    description: str
    validation: Optional[str] = None


@dataclass
class ToolInfo:
    """Represents a FastMCP tool."""

    name: str
    category: str
    description: str
    parameters: List[Parameter]
    returns: str
    examples: List[str]
    notes: List[str]
    validation_model: Optional[str] = None


class ToolDocExtractor:
    """Extracts tool information from Python source code."""

    # Tool categories based on naming patterns
    CATEGORIES = {
        "pick": "Robot Control",
        "place": "Robot Control",
        "push": "Robot Control",
        "move": "Robot Control",
        "calibrate": "Robot Control",
        "clear_collision": "Robot Control",
        "get_detected": "Object Detection",
        "get_largest": "Object Detection",
        "get_smallest": "Object Detection",
        "get_workspace": "Workspace",
        "get_object_labels": "Workspace",
        "add_object": "Workspace",
        "speak": "Feedback",
        "set_user_task": "Feedback",
    }

    def __init__(self, server_file: Path):
        """Initialize extractor with server file path."""
        self.server_file = server_file
        self.source_code = server_file.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source_code)
        self.tools: List[ToolInfo] = []

    def extract_all_tools(self) -> List[ToolInfo]:
        """Extract all @mcp.tool decorated functions."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_mcp_tool(node):
                    tool_info = self._extract_tool_info(node)
                    if tool_info:
                        self.tools.append(tool_info)

        # Sort by category and name
        self.tools.sort(key=lambda t: (t.category, t.name))
        return self.tools

    def _is_mcp_tool(self, node: ast.FunctionDef) -> bool:
        """Check if function is decorated with @mcp.tool."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                if decorator.attr == "tool":
                    return True
            elif isinstance(decorator, ast.Name):
                if decorator.id == "tool":
                    return True
        return False

    def _extract_tool_info(self, node: ast.FunctionDef) -> Optional[ToolInfo]:
        """Extract complete tool information from AST node."""
        try:
            # Get function name
            name = node.name

            # Get category
            category = self._categorize_tool(name)

            # Parse docstring
            docstring = ast.get_docstring(node) or ""
            description, examples, notes = self._parse_docstring(docstring)

            # Extract parameters
            parameters = self._extract_parameters(node)

            # Extract return type
            returns = self._extract_return_type(node)

            # Check for validation decorator
            validation_model = self._get_validation_model(node)

            return ToolInfo(
                name=name,
                category=category,
                description=description,
                parameters=parameters,
                returns=returns,
                examples=examples,
                notes=notes,
                validation_model=validation_model,
            )

        except Exception as e:
            print(f"Warning: Failed to extract {node.name}: {e}")
            return None

    def _categorize_tool(self, name: str) -> str:
        """Determine tool category from name."""
        for pattern, category in self.CATEGORIES.items():
            if pattern in name:
                return category
        return "Other"

    def _parse_docstring(self, docstring: str) -> Tuple[str, List[str], List[str]]:
        """Parse docstring into description, examples, and notes."""
        lines = docstring.split("\n")

        # Extract description (first paragraph)
        description_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                break
            if not any(x in stripped for x in ["Args:", "Returns:", "Example:", "Note:"]):
                description_lines.append(stripped)

        description = " ".join(description_lines)

        # Extract examples
        examples = []
        in_example = False
        example_block = []

        for line in lines:
            if "Example" in line:
                in_example = True
                continue

            if in_example:
                if line.strip() and not line.startswith("    "):
                    # End of example block
                    if example_block:
                        examples.append("\n".join(example_block))
                        example_block = []
                    in_example = False
                else:
                    # Part of example
                    example_block.append(line.rstrip())

        # Add last example if any
        if example_block:
            examples.append("\n".join(example_block))

        # Extract notes
        notes = []
        for i, line in enumerate(lines):
            if "Note:" in line or "IMPORTANT:" in line or "WARNING:" in line:
                # Collect note lines
                note_lines = [line.strip()]
                for next_line in lines[i + 1 :]:
                    if next_line.strip() and not any(x in next_line for x in ["Args:", "Returns:", "Example:"]):
                        note_lines.append(next_line.strip())
                    else:
                        break
                notes.append(" ".join(note_lines))

        return description, examples, notes

    def _extract_parameters(self, node: ast.FunctionDef) -> List[Parameter]:
        """Extract function parameters with type hints and descriptions."""
        parameters = []
        docstring = ast.get_docstring(node) or ""

        # Parse parameter descriptions from docstring
        param_descriptions = self._parse_parameter_descriptions(docstring)

        # Get function arguments
        for arg in node.args.args:
            if arg.arg == "self":
                continue

            # Get type hint
            type_hint = "Any"
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)

            # Get default value
            default = None
            defaults_start = len(node.args.args) - len(node.args.defaults)
            arg_index = node.args.args.index(arg)
            if arg_index >= defaults_start:
                default_node = node.args.defaults[arg_index - defaults_start]
                default = ast.unparse(default_node)

            # Get description from docstring
            description = param_descriptions.get(arg.arg, "")

            # Get validation info from Pydantic model if available
            validation = self._get_parameter_validation(node, arg.arg)

            parameters.append(
                Parameter(name=arg.arg, type_hint=type_hint, default=default, description=description, validation=validation)
            )

        return parameters

    def _parse_parameter_descriptions(self, docstring: str) -> Dict[str, str]:
        """Extract parameter descriptions from docstring."""
        descriptions = {}
        lines = docstring.split("\n")

        in_args_section = False
        current_param = None
        current_desc = []

        for line in lines:
            stripped = line.strip()

            # Check for Args section
            if "Args:" in stripped:
                in_args_section = True
                continue

            # Check for end of Args section
            if in_args_section and stripped and not stripped.startswith(" ") and ":" not in stripped:
                # Save previous parameter
                if current_param and current_desc:
                    descriptions[current_param] = " ".join(current_desc)
                break

            if in_args_section:
                # New parameter
                if stripped and ":" in stripped and not stripped.startswith(" "):
                    # Save previous parameter
                    if current_param and current_desc:
                        descriptions[current_param] = " ".join(current_desc)

                    # Parse new parameter
                    parts = stripped.split(":", 1)
                    if len(parts) == 2:
                        # Extract parameter name (might have type hint)
                        param_part = parts[0].strip()
                        if "(" in param_part:
                            current_param = param_part.split("(")[0].strip()
                        else:
                            current_param = param_part

                        # Start description
                        current_desc = [parts[1].strip()]
                elif current_param and stripped:
                    # Continue description
                    current_desc.append(stripped)

        # Save last parameter
        if current_param and current_desc:
            descriptions[current_param] = " ".join(current_desc)

        return descriptions

    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """Extract return type from function."""
        if node.returns:
            return_type = ast.unparse(node.returns)
        else:
            return_type = "str"

        # Extract return description from docstring
        docstring = ast.get_docstring(node) or ""
        return_desc = ""

        lines = docstring.split("\n")
        in_returns = False

        for line in lines:
            if "Returns:" in line:
                in_returns = True
                # Get description on same line if any
                parts = line.split("Returns:", 1)
                if len(parts) > 1 and parts[1].strip():
                    return_desc = parts[1].strip()
                continue

            if in_returns:
                stripped = line.strip()
                if stripped and not any(x in stripped for x in ["Args:", "Example:", "Note:"]):
                    return_desc += " " + stripped
                else:
                    break

        if return_desc:
            return f"{return_type} - {return_desc.strip()}"
        return return_type

    def _get_validation_model(self, node: ast.FunctionDef) -> Optional[str]:
        """Check if function uses @validate_input decorator and extract model."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    if decorator.func.id == "validate_input":
                        if decorator.args:
                            return ast.unparse(decorator.args[0])
        return None

    def _get_parameter_validation(self, node: ast.FunctionDef, param_name: str) -> Optional[str]:
        """Get validation info for parameter from Pydantic model."""
        validation_model = self._get_validation_model(node)
        if not validation_model:
            return None

        # Try to import and inspect the model
        try:
            # Import schemas module
            import schemas

            model_class = getattr(schemas, validation_model, None)
            if model_class:
                # Get field info
                fields = model_class.model_fields
                if param_name in fields:
                    field = fields[param_name]
                    constraints = []

                    # Extract constraints
                    if hasattr(field, "gt"):
                        constraints.append(f"> {field.gt}")
                    if hasattr(field, "ge"):
                        constraints.append(f">= {field.ge}")
                    if hasattr(field, "lt"):
                        constraints.append(f"< {field.lt}")
                    if hasattr(field, "le"):
                        constraints.append(f"<= {field.le}")
                    if hasattr(field, "min_length"):
                        constraints.append(f"min_length={field.min_length}")
                    if hasattr(field, "max_length"):
                        constraints.append(f"max_length={field.max_length}")

                    if constraints:
                        return ", ".join(constraints)

        except Exception:
            pass

        return None


class MarkdownGenerator:
    """Generates markdown documentation from tool information."""

    def __init__(self, tools: List[ToolInfo]):
        """Initialize with list of tools."""
        self.tools = tools
        self.categories = self._group_by_category()

    def _group_by_category(self) -> Dict[str, List[ToolInfo]]:
        """Group tools by category."""
        categories = {}
        for tool in self.tools:
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool)
        return categories

    def generate(self) -> str:
        """Generate complete markdown documentation."""
        sections = [
            self._generate_header(),
            self._generate_toc(),
            self._generate_overview(),
            self._generate_quick_reference(),
            self._generate_detailed_documentation(),
            self._generate_footer(),
        ]

        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """Generate document header."""
        return f"""# Robot MCP - Auto-Generated API Reference

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Source:** `server/fastmcp_robot_server_unified.py`
**Total Tools:** {len(self.tools)}

> ⚠️ **Note:** This documentation is auto-generated from source code.
> Do not edit manually. Run `python docs/generate_api_docs.py` to update.

---"""

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        lines = ["## Table of Contents\n"]

        # Overview sections
        lines.append("- [Overview](#overview)")
        lines.append("- [Quick Reference](#quick-reference)")

        # Tool categories
        lines.append("- [API Tools](#api-tools)")
        for category in sorted(self.categories.keys()):
            anchor = category.lower().replace(" ", "-")
            count = len(self.categories[category])
            lines.append(f"  - [{category} ({count})](#tools-{anchor})")

        return "\n".join(lines)

    def _generate_overview(self) -> str:
        """Generate overview section."""
        return f"""## Overview

The Robot MCP system provides **{len(self.tools)} tools** organized into **{len(self.categories)} categories**:

{self._generate_category_table()}

### Tool Categories

{self._generate_category_descriptions()}

### Using the API

All tools are exposed via FastMCP and can be called through:

1. **Universal Client** - Multi-LLM support (OpenAI, Groq, Gemini, Ollama)
2. **Direct MCP Client** - Low-level FastMCP protocol
3. **Web GUI** - Gradio interface with voice input
4. **REST API** - HTTP endpoints (if enabled)

See [Setup Guide](mcp_setup_guide.md) for usage instructions."""

    def _generate_category_table(self) -> str:
        """Generate category summary table."""
        lines = ["| Category | Tools | Description |", "|----------|-------|-------------|"]

        category_desc = {
            "Robot Control": "Physical robot manipulation and movement",
            "Object Detection": "Vision-based object recognition and querying",
            "Workspace": "Workspace configuration and coordinate queries",
            "Feedback": "User feedback via speech and text",
            "Other": "Miscellaneous utilities",
        }

        for category in sorted(self.categories.keys()):
            count = len(self.categories[category])
            desc = category_desc.get(category, "Additional tools")
            lines.append(f"| {category} | {count} | {desc} |")

        return "\n".join(lines)

    def _generate_category_descriptions(self) -> str:
        """Generate detailed category descriptions."""
        descriptions = {
            "Robot Control": """**Robot Control** tools provide physical manipulation capabilities:
- Pick and place operations
- Pushing large objects
- Precise movement control
- Calibration and safety""",
            "Object Detection": """**Object Detection** tools use computer vision:
- Real-time object detection via camera
- Spatial filtering (left/right/above/below)
- Size-based sorting and querying
- Label-based filtering""",
            "Workspace": """**Workspace** tools manage coordinate systems:
- Workspace boundary queries
- Free space detection
- Object label management
- Coordinate transformations""",
            "Feedback": """**Feedback** tools provide user communication:
- Text-to-speech output
- Task tracking for video overlays
- Status updates""",
        }

        lines = []
        for category in sorted(self.categories.keys()):
            if category in descriptions:
                lines.append(descriptions[category])

        return "\n\n".join(lines)

    def _generate_quick_reference(self) -> str:
        """Generate quick reference table."""
        lines = [
            "## Quick Reference\n",
            "### All Tools at a Glance\n",
            "| Tool | Category | Input Validation | Description |",
            "|------|----------|------------------|-------------|",
        ]

        for tool in self.tools:
            # Shorten description
            desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
            validation = "✓" if tool.validation_model else "—"
            lines.append(f"| [`{tool.name}`](#{tool.name}) | {tool.category} | {validation} | {desc} |")

        return "\n".join(lines)

    def _generate_detailed_documentation(self) -> str:
        """Generate detailed tool documentation."""
        sections = ["## API Tools\n"]

        for category in sorted(self.categories.keys()):
            sections.append(self._generate_category_section(category))

        return "\n\n---\n\n".join(sections)

    def _generate_category_section(self, category: str) -> str:
        """Generate documentation for a category."""
        tools = self.categories[category]
        category.lower().replace(" ", "-")

        lines = [f"### Tools: {category}\n"]

        for tool in tools:
            lines.append(self._generate_tool_section(tool))
            lines.append("\n---\n")

        return "\n".join(lines)

    def _generate_tool_section(self, tool: ToolInfo) -> str:
        """Generate documentation for a single tool."""
        sections = [
            f"#### {tool.name}\n",
            f"**Category:** {tool.category}  ",
        ]

        if tool.validation_model:
            sections.append(f"**Validation:** `{tool.validation_model}`  ")

        sections.append(f"\n{tool.description}\n")

        # Function signature
        sections.append("**Signature:**\n```python")
        sections.append(self._generate_signature(tool))
        sections.append("```\n")

        # Parameters
        if tool.parameters:
            sections.append("**Parameters:**\n")
            for param in tool.parameters:
                sections.append(self._format_parameter(param))

        # Returns
        sections.append("**Returns:**  ")
        sections.append(f"`{tool.returns}`\n")

        # Examples
        if tool.examples:
            sections.append("**Examples:**\n")
            for example in tool.examples:
                sections.append("```python")
                sections.append(example.strip())
                sections.append("```\n")

        # Notes
        if tool.notes:
            sections.append("**Notes:**\n")
            for note in tool.notes:
                sections.append(f"- {note}")

        return "\n".join(sections)

    def _generate_signature(self, tool: ToolInfo) -> str:
        """Generate function signature."""
        params = []
        for param in tool.parameters:
            if param.default:
                params.append(f"{param.name}: {param.type_hint} = {param.default}")
            else:
                params.append(f"{param.name}: {param.type_hint}")

        return f"def {tool.name}({', '.join(params)}) -> {tool.returns.split('-')[0].strip()}"

    def _format_parameter(self, param: Parameter) -> str:
        """Format a parameter for documentation."""
        lines = [f"- `{param.name}` (`{param.type_hint}`"]

        if param.default:
            lines.append(f", default: `{param.default}`")

        lines.append(")")

        if param.description:
            lines.append(f": {param.description}")

        if param.validation:
            lines.append(f"  - **Validation:** {param.validation}")

        return "".join(lines) + "\n"

    def _generate_footer(self) -> str:
        """Generate document footer."""
        return """---

## Additional Resources

- **[Setup Guide](mcp_setup_guide.md)** - Installation and configuration
- **[Examples](examples.md)** - Common use cases and workflows
- **[Architecture](README.md)** - System design and data flow
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Contributing

To update this documentation:

1. Modify tool docstrings in `server/fastmcp_robot_server_unified.py`
2. Run: `python docs/generate_api_docs.py`
3. Commit both source and generated docs

## Validation

All tools with `@validate_input` decorator use Pydantic models for input validation.
See `server/schemas.py` for validation model definitions.

---

**Auto-generated by:** `docs/generate_api_docs.py`
**Last updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""


class DocumentationValidator:
    """Validates completeness and quality of documentation."""

    def __init__(self, tools: List[ToolInfo]):
        """Initialize validator with tools."""
        self.tools = tools
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Run all validation checks."""
        self._check_completeness()
        self._check_quality()
        self._check_consistency()
        return len(self.issues) == 0

    def _check_completeness(self):
        """Check if all tools have required documentation."""
        for tool in self.tools:
            # Check description
            if not tool.description or len(tool.description) < 20:
                self.issues.append(f"{tool.name}: Description too short or missing")

            # Check parameters
            for param in tool.parameters:
                if not param.description:
                    self.warnings.append(f"{tool.name}.{param.name}: Missing parameter description")

            # Check examples
            if not tool.examples:
                self.warnings.append(f"{tool.name}: No usage examples provided")

    def _check_quality(self):
        """Check documentation quality."""
        for tool in self.tools:
            # Check description clarity
            if tool.description:
                # Avoid generic descriptions
                generic_phrases = ["This function", "This method", "This tool"]
                if any(phrase in tool.description for phrase in generic_phrases):
                    self.warnings.append(f"{tool.name}: Description uses generic phrasing")

            # Check example quality
            for i, example in enumerate(tool.examples):
                if not example.strip():
                    self.warnings.append(f"{tool.name}: Example {i+1} is empty")
                elif len(example) < 10:
                    self.warnings.append(f"{tool.name}: Example {i+1} is too short")

    def _check_consistency(self):
        """Check consistency across documentation."""
        # Check return type consistency
        return_types = {}
        for tool in self.tools:
            if "str" in tool.returns:
                return_types[tool.name] = "str"

        # Most tools should return str
        if len(return_types) < len(self.tools) * 0.8:
            self.warnings.append("Inconsistent return types across tools")

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 60)
        print("DOCUMENTATION VALIDATION REPORT")
        print("=" * 60)

        if not self.issues and not self.warnings:
            print("✓ All documentation checks passed!")
            return

        if self.issues:
            print(f"\n❌ Critical Issues ({len(self.issues)}):")
            for issue in self.issues[:20]:
                print(f"  • {issue}")
            if len(self.issues) > 20:
                print(f"  ... and {len(self.issues) - 20} more issues")

        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:20]:
                print(f"  • {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")

        # Summary
        total = len(self.tools)
        complete = sum(1 for t in self.tools if t.description and t.examples)
        print("\n📊 Summary:")
        print(f"  Total tools: {total}")
        print(f"  Fully documented: {complete} ({complete/total*100:.1f}%)")
        print(f"  Issues: {len(self.issues)}")
        print(f"  Warnings: {len(self.warnings)}")


def main():
    """Main entry point."""
    # Paths
    repo_root = Path(__file__).parent.parent
    server_file = repo_root / "server" / "fastmcp_robot_server_unified.py"
    output_file = repo_root / "docs" / "api_reference_auto.md"

    print("=" * 60)
    print("Robot MCP - API Documentation Generator")
    print("=" * 60)
    print(f"Source: {server_file}")
    print(f"Output: {output_file}")
    print()

    # Check if server file exists
    if not server_file.exists():
        print(f"❌ Error: Server file not found: {server_file}")
        sys.exit(1)

    # Extract tools
    print("Extracting tool information...")
    extractor = ToolDocExtractor(server_file)
    tools = extractor.extract_all_tools()

    print(f"✓ Found {len(tools)} tools")

    # Group by category
    categories = {}
    for tool in tools:
        if tool.category not in categories:
            categories[tool.category] = 0
        categories[tool.category] += 1

    print("\nTools by category:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    # Generate documentation
    print("\nGenerating markdown documentation...")
    generator = MarkdownGenerator(tools)
    markdown = generator.generate()

    # Write to file
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(markdown, encoding="utf-8")

    print(f"✓ Documentation generated: {output_file}")
    print(f"  Size: {len(markdown):,} characters")
    print(f"  Lines: {markdown.count(chr(10)):,}")

    # Validation
    print("\nValidating documentation...")
    validator = DocumentationValidator(tools)
    is_valid = validator.validate()
    validator.print_report()

    if not is_valid:
        print("\n❌ Documentation validation failed!")
        print("   Fix critical issues before committing.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Documentation generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

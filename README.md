# Robot MCP Control System

Natural language robot control using **FastMCP** and Groq's LLM API.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/robot_mcp/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/robot_mcp/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/robot_mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/robot_mcp/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/robot_mcp/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/robot_mcp/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 🎯 Overview

Control robotic arms (Niryo Ned2, WidowX) through natural language using the Model Context Protocol (MCP) and large language models. Simply tell the robot what to do: *"Pick up the pencil and place it next to the red cube"*.

## 🎯 How It Works

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│             │  MCP    │              │  Robot  │             │
│  Groq LLM   │◄───────►│  MCP Server  │◄───────►│   Niryo/    │
│   Client    │Protocol │   (FastMCP)  │   API   │   WidowX    │
│             │         │              │         │             │
└─────────────┘         └──────────────┘         └─────────────┘
      ▲                                                 │
      │                                                 │
      │ Natural Language                     Physical   │
      │ Commands                             Actions    │
      │                                                 │
   ┌──┴──┐                                          ┌───▼───┐
   │User │                                          │Objects│
   └─────┘                                          └───────┘
```

### System Flow

1. **User** speaks natural language: "Pick up the pencil"
2. **Groq LLM** interprets command and decides which tools to call
3. **MCP Client** sends tool calls to MCP server via SSE
4. **MCP Server** executes robot commands
5. **Robot** performs physical actions
6. **Results** flow back to user through the chain

## ✨ Key Features

- 🤖 **Natural Language Control** - No programming required
- 🔧 **Multi-Robot Support** - Niryo Ned2 and WidowX
- 👁️ **Vision-Based Detection** - Automatic object detection and tracking
- 🎨 **Gradio Web Interface** - User-friendly GUI with live camera feed
- 🎤 **Voice Input** - Speak commands directly to the robot
- 🔊 **Audio Feedback** - Robot speaks status updates
- 🎯 **Spatial Reasoning** - Understands relative positions and arrangements

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- Niryo Ned2 or WidowX robot (or simulation)
- Groq API key ([Get one free](https://console.groq.com/keys))

### Installation

```bash
# Clone repository
git clone https://github.com/dgaida/robot_mcp.git
cd robot_mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Configure API keys
cp secrets.env.template secrets.env
# Edit secrets.env and add your GROQ_API_KEY
```

### Start Redis

```bash
docker run -p 6379:6379 redis:alpine
```

### Quick Test

```bash
# Terminal 1: Start FastMCP Server
python server/fastmcp_robot_server.py --robot niryo --no-simulation

# Terminal 2: Run interactive client
python -c "from client.fastmcp_groq_client import RobotFastMCPClient; import asyncio; asyncio.run(RobotFastMCPClient('your_groq_key').connect())"
```

## 💻 Usage

### Interactive Chat Mode

```bash
python fastmcp_main_client.py
```

Example conversation:

```
You: What objects do you see?
🤖: I can see 3 objects: a pencil at [0.15, -0.05], 
    a red cube at [0.20, 0.10], and a blue square at [0.18, -0.10]

You: Move the pencil next to the red cube
🤖: Done! I've placed the pencil to the right of the red cube.

You: Arrange all objects in a line
🤖: All objects are now arranged in a horizontal line, 
    spaced 8cm apart.
```

### Web GUI

```bash
# Launch GUI
./launch_gui.sh  # Linux/Mac
# or
launch_gui.bat   # Windows

# Options:
./launch_gui.sh --robot widowx --real --share
```

### Programmatic Usage

```python
from client.fastmcp_groq_client import RobotFastMCPClient
import asyncio

async def demo():
    client = RobotFastMCPClient(
        groq_api_key="your_key",
        model="moonshotai/kimi-k2-instruct-0905"
    )
    
    await client.connect()
    
    # Natural language commands
    await client.chat("What objects do you see?")
    await client.chat("Pick up the largest object and place it in the center")
    await client.chat("Sort all objects by size")
    
    await client.disconnect()

asyncio.run(demo())
```

## 🛠️ Available Tools

The FastMCP server exposes these robot control tools:

### Robot Control
- `pick_place_object` - Complete pick and place operation
- `pick_object` - Pick up an object
- `place_object` - Place a held object
- `push_object` - Push objects (for items too large to grip)
- `move2observation_pose` - Position for workspace observation

### Object Detection
- `get_detected_objects` - List all detected objects
- `get_detected_object` - Find object at coordinates
- `get_largest_detected_object` - Get biggest object
- `get_smallest_detected_object` - Get smallest object
- `get_detected_objects_sorted` - Sort objects by size

### Workspace
- `get_largest_free_space_with_center` - Find free space for placement
- `get_workspace_coordinate_from_point` - Get corner/center coordinates
- `get_object_labels_as_string` - List recognizable objects
- `add_object_name2object_labels` - Add new object type

### Feedback
- `speak` - Text-to-speech output

## 📚 Example Tasks

### Simple Commands
```
"What objects do you see?"
"Pick up the pencil and place it at [0.2, 0.1]"
"Move the red cube to the left of the blue square"
"Show me the largest object"
```

### Advanced Tasks
```
"Sort all objects by size from smallest to largest"
"Arrange objects in a triangle pattern"
"Group objects by color: red on left, blue on right"
"Swap positions of the two largest objects"
"Pick all pens and place them next to the chocolate bar"
```

### Complex Workflows
```
"Execute: 1) Find all objects 2) Move smallest to [0.15, 0.1] 
3) Move largest right of smallest 4) Report positions"

"Organize the workspace: cubes on left, cylinders in middle, 
everything else on right, aligned in rows"
```

## 🎮 Gradio Web Interface

The web GUI provides:

- 💬 **Chat Interface** - Natural language interaction
- 📹 **Live Camera** - Real-time workspace view with object annotations
- 🎤 **Voice Input** - Speak your commands
- 📊 **System Status** - Connection and operation monitoring
- 📝 **Example Tasks** - Quick-start templates

Launch with:
```bash
python robot_gui/mcp_app.py --robot niryo
```

## ⚙️ Configuration

### Environment Variables

Create `secrets.env`:

```bash
GROQ_API_KEY=gsk_your_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key  # Optional for TTS
```

### FastMCP Server Options

```bash
python server/fastmcp_robot_server.py \
  --robot niryo \              # or widowx
  --no-simulation \            # Use real robot
  --host 127.0.0.1 \
  --port 8000 \
  --verbose
```

### Groq Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| moonshotai/kimi-k2-instruct-0905 | Very Fast | Excellent | Default choice |
| llama-3.3-70b-versatile | Fast | Excellent | Complex tasks |
| llama-3.1-8b-instant | Very Fast | Good | Simple commands |

## 📖 Documentation

- **[Architecture Guide](docs/README.md)** - System design and data flow
- **[API Reference](docs/api.md)** - Complete tool documentation
- **[Examples](docs/examples.md)** - Common use cases
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black .
ruff check .
mypy robot_mcp/
```

### Adding New Tools

1. Add tool definition in `server/fastmcp_robot_server.py`:

```python
@mcp.tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """Tool description."""
    result = # Your implementation
    return result
```

2. The tool is automatically available to the LLM client

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Update documentation
5. Submit a pull request

## 📝 Notes

**MCP vs FastMCP**: This repository now uses **FastMCP** exclusively. The older MCP implementation (stdio-based) is deprecated but still present in the codebase for reference.

**Dependencies**: [Robot Environment](https://github.com/dgaida/robot_environment) and [Text2Speech](https://github.com/dgaida/text2speech) are automatically installed from GitHub as dependencies.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) - Communication framework
- [Groq](https://groq.com) - Fast LLM inference
- [FastMCP](https://github.com/jlowin/fastmcp) - Modern MCP implementation
- [Niryo Robotics](https://niryo.com) - Robot hardware
- [OwlV2](https://huggingface.co/google/owlv2-base-patch16-ensemble) - Object detection

## 📧 Contact

Daniel Gaida - daniel.gaida@th-koeln.de

Project Link: [https://github.com/dgaida/robot_mcp](https://github.com/dgaida/robot_mcp)

---

*Made with ❤️ for robotic automation*

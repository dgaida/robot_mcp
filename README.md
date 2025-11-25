# Robot MCP Control System

Natural language robot control using **FastMCP** and **Multi-LLM Support** (OpenAI, Groq, Gemini, Ollama).

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/robot_mcp/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/robot_mcp)
[![Code Quality](https://github.com/dgaida/robot_mcp/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/robot_mcp/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/robot_mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/robot_mcp/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/robot_mcp/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/robot_mcp/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 🎯 Overview

Control robotic arms (Niryo Ned2, WidowX) through natural language using the Model Context Protocol (MCP) and **multiple LLM providers**. Simply tell the robot what to do: *"Pick up the pencil and place it next to the red cube"*.

### 🆕 Multi-LLM Support

Now supports **4 LLM providers** with automatic API detection:

| Provider | Models | Best For | Speed |
|----------|--------|----------|-------|
| **OpenAI** | GPT-4o, GPT-4o-mini | Complex reasoning | Fast |
| **Groq** | Kimi K2, Llama 3.3, Mixtral | Ultra-fast inference | Very Fast |
| **Google Gemini** | Gemini 2.0/2.5 | Long context, multimodal | Fast |
| **Ollama** | Llama 3.2, Mistral, CodeLlama | Local/offline use | Variable |

## 🎯 How It Works

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Multi-    │  MCP    │              │  Robot  │             │
│   LLM       │◄───────►│  MCP Server  │◄───────►│   Niryo/    │
│  (OpenAI/   │Protocol │   (FastMCP)  │   API   │   WidowX    │
│ Groq/Gemini)│         │              │         │             │
└─────────────┘         └──────────────┘         └─────────────┘
      ▲                                                 │
      │ Natural Language                     Physical   │
      │ Commands                             Actions    │
   ┌──┴──┐                                          ┌───▼───┐
   │User │                                          │Objects│
   └─────┘                                          └───────┘
```

### System Flow

1. **User** speaks natural language: "Pick up the pencil"
2. **LLM** interprets command and decides which tools to call
3. **MCP Client** sends tool calls to MCP server via SSE
4. **MCP Server** executes robot commands
5. **Robot** performs physical actions
6. **Results** flow back to user through the chain

## ✨ Key Features

- 🤖 **Natural Language Control** - No programming required
- 🔧 **Multi-LLM Support** - Choose OpenAI, Groq, Gemini, or Ollama
- 🎯 **Auto-Detection** - Automatically selects available API
- 🔄 **Hot-Swapping** - Switch providers during runtime
- 🤖 **Multi-Robot Support** - Niryo Ned2 and WidowX
- 👁️ **Vision-Based Detection** - Automatic object detection
- 🎨 **Gradio Web Interface** - User-friendly GUI
- 🎤 **Voice Input** - Speak commands directly
- 🔊 **Audio Feedback** - Robot speaks status updates

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- Niryo Ned2 or WidowX robot (or simulation)
- **At least one API key** from:
  - [OpenAI](https://platform.openai.com/api-keys) (GPT-4o, GPT-4o-mini)
  - [Groq](https://console.groq.com/keys) (Free tier available)
  - [Google AI Studio](https://aistudio.google.com/apikey) (Gemini)
  - [Ollama](https://ollama.com/) (Local, no API key needed)

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
# Edit secrets.env and add your API key(s)
```

### Configure API Keys

Edit `secrets.env`:

```bash
# Add at least one API key (priority: OpenAI > Groq > Gemini)

# OpenAI (GPT-4o, GPT-4o-mini)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# Groq (Kimi, Llama, Mixtral) - Free tier available!
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxx

# Google Gemini (Gemini 2.0, 2.5)
GEMINI_API_KEY=AIzaSy-xxxxxxxxxxxxxxxx

# Ollama - No API key needed (runs locally)
# Just install: curl -fsSL https://ollama.ai/install.sh | sh
```

### Start Redis

```bash
docker run -p 6379:6379 redis:alpine
```

### Quick Test

```bash
# Terminal 1: Start FastMCP Server
python server/fastmcp_robot_server.py --robot niryo --no-simulation

# Terminal 2: Run universal client (auto-detects available API)
python client/fastmcp_universal_client.py
```

## 💻 Usage

### Universal Client (Recommended)

The new universal client auto-detects and uses available LLM providers:

```bash
# Auto-detect API (uses first available: OpenAI > Groq > Gemini > Ollama)
python client/fastmcp_universal_client.py

# Explicitly use OpenAI
python client/fastmcp_universal_client.py --api openai --model gpt-4o

# Use Groq (fastest inference)
python client/fastmcp_universal_client.py --api groq

# Use Gemini
python client/fastmcp_universal_client.py --api gemini --model gemini-2.0-flash

# Use local Ollama (no internet required)
python client/fastmcp_universal_client.py --api ollama --model llama3.2:1b

# Single command mode
python client/fastmcp_universal_client.py --command "What objects do you see?"
```

### Interactive Features

```
You: What objects do you see?
🤖: I can see 3 objects: a pencil at [0.15, -0.05],
    a red cube at [0.20, 0.10], and a blue square at [0.18, -0.10]

You: switch
🔄 Current provider: GROQ
Available: openai, groq, gemini, ollama
Switch to: openai
✓ Switched to OPENAI - gpt-4o-mini

You: Move the pencil next to the red cube
🤖: Done! I've placed the pencil to the right of the red cube.
```

### Legacy Groq-Only Client

The original Groq-specific client is still available:

```bash
python client/fastmcp_groq_client.py
```

### Programmatic Usage

```python
from client.fastmcp_universal_client import RobotUniversalMCPClient
import asyncio

async def demo():
    # Auto-detect available API
    client = RobotUniversalMCPClient()

    # Or specify provider
    # client = RobotUniversalMCPClient(api_choice="openai", model="gpt-4o")

    await client.connect()

    # Natural language commands work with any provider
    await client.chat("What objects do you see?")
    await client.chat("Pick up the largest object")
    await client.chat("Place it in the center")

    await client.disconnect()

asyncio.run(demo())
```

## 🛠️ Available Tools

The FastMCP server exposes these robot control tools (work with all LLM providers):

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

## 📊 LLM Provider Comparison

### Performance Characteristics

| Provider | Function Calling | Speed | Cost | Offline | Best Use Case |
|----------|-----------------|-------|------|---------|---------------|
| **OpenAI** | ✅ Excellent | Fast | $$ | ❌ | Production, complex tasks |
| **Groq** | ✅ Excellent | Very Fast | Free tier | ❌ | Development, prototyping |
| **Gemini** | ✅ Excellent | Fast | Free tier | ❌ | Long context, multimodal |
| **Ollama** | ⚠️ Limited | Variable | Free | ✅ | Local testing, privacy |

### Recommended Models

**For Complex Tasks:**
```bash
# OpenAI - Best reasoning
--api openai --model gpt-4o

# Groq - Fastest inference
--api groq --model moonshotai/kimi-k2-instruct-0905
```

**For Development:**
```bash
# OpenAI - Fast and cheap
--api openai --model gpt-4o-mini

# Groq - Free and fast
--api groq --model llama-3.3-70b-versatile
```

**For Local/Offline:**
```bash
# Ollama - No internet required
--api ollama --model llama3.2:1b
```

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
```

### Complex Workflows
```
"Execute: 1) Find all objects 2) Move smallest to [0.15, 0.1]
3) Move largest right of smallest 4) Report positions"

"Organize the workspace: cubes on left, cylinders in middle,
everything else on right, aligned in rows"
```

## 🎮 Gradio Web Interface

The web GUI supports all LLM providers:

```bash
python robot_gui/mcp_app.py --robot niryo
```

Features:
- 💬 Chat with robot using any LLM provider
- 📹 Live camera feed with object annotations
- 🎤 Voice input (Whisper)
- 📊 System status monitoring
- 🔄 Switch LLM providers on-the-fly

## ⚙️ Configuration

### Environment Variables

Create `secrets.env`:

```bash
# Multi-LLM Support - Add any/all of these

# OpenAI (priority if multiple keys present)
OPENAI_API_KEY=sk-xxxxxxxx

# Groq (fast, free tier available)
GROQ_API_KEY=gsk-xxxxxxxx

# Google Gemini
GEMINI_API_KEY=AIzaSy-xxxxxxxx

# Optional: ElevenLabs for better TTS
ELEVENLABS_API_KEY=your_key
```

### API Priority

If multiple API keys are present, the client uses this priority:
1. **OpenAI** (if `OPENAI_API_KEY` set)
2. **Groq** (if `GROQ_API_KEY` set)
3. **Gemini** (if `GEMINI_API_KEY` set)
4. **Ollama** (fallback, no key needed)

Override with `--api` flag:
```bash
# Force Gemini even if OpenAI key exists
python client/fastmcp_universal_client.py --api gemini
```

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

## 📖 Documentation

- **[Architecture Guide](docs/README.md)** - System design and data flow
- **[API Reference](docs/api.md)** - Complete tool documentation
- **[Examples](docs/examples.md)** - Common use cases
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Update documentation
5. Submit a pull request

## 📝 Notes

**Multi-LLM Architecture**: This repository uses the `LLMClient` class from the [llm_client](https://github.com/dgaida/llm_client) repository, providing unified access to multiple LLM providers.

**Function Calling Support**: OpenAI, Groq, and Gemini all support function calling natively. Ollama has limited support and falls back to text-based instruction following.

**Dependencies**: [Robot Environment](https://github.com/dgaida/robot_environment) and [Text2Speech](https://github.com/dgaida/text2speech) are automatically installed from GitHub.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) - Communication framework
- [OpenAI](https://openai.com) - GPT models
- [Groq](https://groq.com) - Fast LLM inference
- [Google Gemini](https://ai.google.dev/gemini-api) - Gemini models
- [Ollama](https://ollama.com) - Local LLM deployment
- [FastMCP](https://github.com/jlowin/fastmcp) - Modern MCP implementation
- [Niryo Robotics](https://niryo.com) - Robot hardware

## 📧 Contact

Daniel Gaida - daniel.gaida@th-koeln.de

Project Link: [https://github.com/dgaida/robot_mcp](https://github.com/dgaida/robot_mcp)

---

*Made with ❤️ for robotic automation*

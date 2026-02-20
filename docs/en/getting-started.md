# MCP Robot Control - Setup & Usage Guide

Complete guide for setting up and using natural language robot control with FastMCP and multi-LLM support.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Modes](#usage-modes)
- [Available LLM Providers](#available-llm-providers)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Robot MCP system enables natural language control of robotic arms (Niryo Ned2, WidowX) using:

- **FastMCP Server** - Exposes robot control tools via HTTP/SSE
- **Universal Client** - Supports OpenAI, Groq, Gemini, and Ollama
- **Vision System** - Real-time object detection
- **Web Interface** - Gradio GUI with voice input

### System Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Multi-    │  HTTP   │              │  Python │             │
│   LLM       │◄───────►│ FastMCP      │◄───────►│   Niryo/    │
│  (OpenAI/   │  SSE    │ Server       │   API   │   WidowX    │
│ Groq/Gemini)│         │              │         │             │
└─────────────┘         └──────────────┘         └─────────────┘
      ▲                                                 │
      │ Natural Language                     Physical   │
      │ Commands                             Actions    │
   ┌──┴──┐                                          ┌───▼───┐
   │User │                                          │Objects│
   └─────┘                                          └───────┘
```

---

## Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.8+
- Redis server
- Niryo Ned2 or WidowX robot (or simulation)
- At least one LLM API key (OpenAI, Groq, or Gemini) OR Ollama installed
```

### 3-Step Setup

**Step 1: Install Dependencies**

```bash
git clone https://github.com/dgaida/robot_mcp.git
cd robot_mcp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

**Step 2: Configure API Keys**

```bash
cp secrets.env.template secrets.env
# Edit secrets.env and add at least one API key:
```

```bash
# OpenAI (best reasoning)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# Groq (fastest, free tier available)
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxx

# Google Gemini (long context)
GEMINI_API_KEY=AIzaSy-xxxxxxxxxxxxxxxx

# Ollama - No API key needed (runs locally)
# Just install: curl -fsSL https://ollama.ai/install.sh | sh
```

**Step 3: Start System**

```bash
# Terminal 1: Start Redis
docker run -p 6379:6379 redis:alpine

# Terminal 2: Start FastMCP Server
python server/fastmcp_robot_server.py --robot niryo

# Terminal 3: Run Universal Client (auto-detects available API)
python client/fastmcp_universal_client.py
```

**You're ready!** The client will automatically use the first available LLM provider (priority: OpenAI > Groq > Gemini > Ollama).

---

## Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/dgaida/robot_mcp.git
cd robot_mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .
```

### Dependencies Included

- `fastmcp` - Modern MCP implementation
- `openai` - OpenAI API client
- `groq` - Groq API client
- `google-generativeai` - Gemini API client
- `ollama` - Local LLM support
- `robot-environment` - Robot control (from GitHub)
- `text2speech` - TTS integration (from GitHub)
- `gradio` - Web interface

---

## Configuration

### Environment Variables

Create `secrets.env` in project root:

```bash
# LLM API Keys (add at least one)
OPENAI_API_KEY=sk-proj-xxxxxxxx
GROQ_API_KEY=gsk-xxxxxxxx
GEMINI_API_KEY=AIzaSy-xxxxxxxx

# Optional: ElevenLabs for better TTS
ELEVENLABS_API_KEY=your_key_here
```

### Server Configuration

**Start server with options:**

```bash
# Real Niryo robot
python server/fastmcp_robot_server.py --robot niryo --no-simulation

# Simulated robot
python server/fastmcp_robot_server.py --robot niryo

# WidowX robot
python server/fastmcp_robot_server.py --robot widowx --no-simulation

# Custom host/port
python server/fastmcp_robot_server.py --host 0.0.0.0 --port 8080

# Disable camera (testing)
python server/fastmcp_robot_server.py --no-camera

# Verbose logging
python server/fastmcp_robot_server.py --verbose
```

### Client Configuration

**Universal client with auto-detection:**

```bash
# Auto-detect API (prefers OpenAI > Groq > Gemini > Ollama)
python client/fastmcp_universal_client.py

# Force specific provider
python client/fastmcp_universal_client.py --api openai --model gpt-4o
python client/fastmcp_universal_client.py --api groq
python client/fastmcp_universal_client.py --api gemini --model gemini-2.0-flash
python client/fastmcp_universal_client.py --api ollama --model llama3.2:1b

# Single command mode
python client/fastmcp_universal_client.py --command "What objects do you see?"

# Adjust parameters
python client/fastmcp_universal_client.py --temperature 0.5 --max-tokens 2048
```

---

## 📚 Usage Modes

### 1. Interactive Chat Mode (Default)

**Best for:** Exploration, learning, development

```bash
python client/fastmcp_universal_client.py

🤖 ROBOT CONTROL ASSISTANT (Universal LLM)
Using: OPENAI - gpt-4o-mini

You: What objects do you see?
🔧 Calling tool: get_detected_objects
✓ Result: Detected 3 objects...

🤖 Assistant: I can see 3 objects:
   1. A pencil at coordinates [0.15, -0.05]
   2. A red cube at [0.20, 0.10]
   3. A blue square at [0.18, -0.10]

You: Move the pencil next to the red cube
🔧 Calling tool: pick_place_object
✓ Result: Successfully picked and placed

🤖 Assistant: Done! I've placed the pencil to the right of the red cube.

# Special commands:
You: tools          # List available tools
You: clear          # Clear conversation history
You: switch         # Switch LLM provider
You: quit           # Exit
```

### 2. Single Command Mode

**Best for:** Scripting, automation, testing

```bash
# Execute one command
python client/fastmcp_universal_client.py --command "Sort objects by size"

# Batch script
#!/bin/bash
commands=(
  "What objects do you see?"
  "Move the largest object to [0.2, 0.0]"
  "Arrange all objects in a line"
)

for cmd in "${commands[@]}"; do
  python client/fastmcp_universal_client.py --command "$cmd"
  sleep 2
done
```

### 3. Gradio Web Interface

**Best for:** User-friendly interaction, demonstrations

```bash
python robot_gui/mcp_app.py --robot niryo

# Then open browser to http://localhost:7860
```

**Features:**
- 💬 Chat interface with robot
- 📹 Live camera feed with object annotations
- 🎤 Voice input (Whisper-based)
- 📊 System status monitoring
- 🔄 Switch LLM providers on-the-fly

### 4. Example Scripts

**Best for:** Learning, templates

```bash
# Run specific example
python examples/universal_examples.py workspace_scan

# Run all examples
python examples/universal_examples.py all

# Compare LLM providers
python examples/universal_examples.py compare_providers
```

### 5. Claude Desktop Integration

**Best for**: Using with Claude's interface

```bash
TODO - not yet implemented
```

Add to Claude Desktop, restart, and use tools directly in Claude!

---

## Available LLM Providers

### Provider Comparison

| Provider | Function Calling | Speed | Cost | Offline | Best For |
|----------|-----------------|-------|------|---------|----------|
| **OpenAI** | ✅ Excellent | Fast | $$ | ❌ | Production, complex reasoning |
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

### Provider Auto-Detection

If you have multiple API keys configured, the client uses this priority:

1. **OpenAI** (if `OPENAI_API_KEY` set)
2. **Groq** (if `GROQ_API_KEY` set)
3. **Gemini** (if `GEMINI_API_KEY` set)
4. **Ollama** (fallback, no key needed)

Override with `--api` flag:
```bash
# Force Gemini even if OpenAI key exists
python client/fastmcp_universal_client.py --api gemini
```

### Switching Providers

**During Interactive Session:**
```
You: switch
🔄 Current provider: GROQ
Available: openai, groq, gemini, ollama
Switch to: openai
✓ Switched to OPENAI - gpt-4o-mini
```

---

## Common Tasks

### Basic Operations

**1. Scan Workspace**
```
You: What objects do you see?
```

**2. Simple Pick and Place**
```
You: Pick up the pencil and place it at [0.2, 0.1]
```

**3. Relative Placement**
```
You: Move the red cube to the right of the blue square
```

### Advanced Tasks

**4. Sort by Size**
```
You: Sort all objects by size from smallest to largest in a line
```

**5. Create Patterns**
```
You: Arrange objects in a triangle pattern
```

**6. Group by Color**
```
You: Group objects by color: red on left, blue on right
```

### Complex Workflows

**7. Multi-Step Task**
```
You: Execute: 1) Find all objects 2) Move smallest to [0.15, 0.1]
     3) Move largest right of smallest 4) Report positions
```

**8. Conditional Logic**
```
You: If there are more than 3 objects, arrange them in a grid.
     Otherwise, arrange them in a line.
```

**9. Workspace Cleanup**
```
You: Organize the workspace: cubes on left, cylinders in middle,
     everything else on right, aligned in rows
```

---

## Troubleshooting

### Server Won't Start

**Problem:** Port 8000 already in use

```bash
# Check what's using the port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

**Problem:** Redis connection error

```bash
# Start Redis
docker run -p 6379:6379 redis:alpine

# Or install locally
# Linux: sudo apt install redis-server
# Mac: brew install redis
```

### Client Can't Connect

**Problem:** "Connection refused"

**Solutions:**
1. Verify server is running:
   ```bash
   curl http://127.0.0.1:8000/sse
   ```

2. Check firewall settings

3. Ensure server started successfully (check logs in `log/` directory)

### API Key Issues

**Problem:** "Invalid API key"

**Solutions:**
1. Verify API key in `secrets.env`:
   ```bash
   cat secrets.env | grep API_KEY
   ```

2. Test API key directly:
   ```python
   from openai import OpenAI
   client = OpenAI(api_key="your_key")
   # Should not raise error
   ```

3. Regenerate key:
   - OpenAI: https://platform.openai.com/api-keys
   - Groq: https://console.groq.com/keys
   - Gemini: https://aistudio.google.com/apikey

### LLM Not Calling Tools

**Problem:** LLM responds in text only, no robot actions

**Solutions:**
1. Verify tools are registered:
   ```
   You: tools
   # Should list: pick_place_object, get_detected_objects, etc.
   ```

2. Use specific commands:
   ```
   ✅ Good: "Pick up the pencil at [0.15, -0.05]"
   ❌ Bad: "Do something with the pencil"
   ```

3. Try different model (some better at tool calling):
   ```bash
   --model gpt-4o  # Better than gpt-3.5-turbo
   ```

### No Objects Detected

**Problem:** `get_detected_objects()` returns empty

**Solutions:**
1. Move to observation pose first:
   ```
   You: Move to observation pose
   ```

2. Check camera:
   - Is camera connected?
   - Is Redis running?
   - Check camera feed in Gradio GUI

3. Verify lighting and object visibility

4. Check object labels:
   ```
   You: What objects can you recognize?
   ```

### Slow Performance

**Problem:** Long response times

**Solutions:**
1. Use faster model:
   ```bash
   --api groq --model llama-3.1-8b-instant
   ```

2. Clear conversation history:
   ```
   You: clear
   ```

3. Reduce detection frequency (edit server):
   ```python
   time.sleep(1.0)  # In camera loop
   ```

### Common Error Messages

```
Error: "Maximum iterations reached"
→ Task too complex, break into smaller steps

Error: "Object not found"
→ Verify object name matches detection exactly (case-sensitive)

Error: "Coordinates out of bounds"
→ Valid range: X=[0.163, 0.337], Y=[-0.087, 0.087]

Error: "Rate limit exceeded"
→ Wait 60 seconds or upgrade API plan
```

---

## Best Practices

### 1. Always Detect Before Manipulating

```
✅ Good: First ask "What objects do you see?"
         Then use coordinates from detection

❌ Bad: Assuming coordinates without checking
```

### 2. Use Exact Label Matching

```python
✅ Good: "pencil" (exact match)
❌ Bad: "Pencil", "PENCIL", "pen" (won't match)
```

### 3. Provide Clear Instructions

```
✅ Good: "Pick up the pencil at [0.15, -0.05] and place it at [0.2, 0.1]"
❌ Bad: "Move that thing over there"
```

### 4. Check for Success

```
✅ Good: After action, ask "Did that work?" or "Show me the result"
❌ Bad: Assuming success without verification
```

### 5. Use Safe Placement

```
✅ Good: "Place object in a safe location" (LLM will find free space)
❌ Bad: Hard-coded coordinates that might collide
```

---

## Quick Reference

### Essential Commands

```bash
# Start system
docker run -p 6379:6379 redis:alpine
python server/fastmcp_robot_server.py --robot niryo
python client/fastmcp_universal_client.py

# Test connection
curl http://127.0.0.1:8000/sse

# View logs
tail -f log/mcp_server_*.log

# Stop server
# Press Ctrl+C in server terminal
```

### Natural Language Examples

```
"What objects do you see?"
"Pick up the pencil"
"Move the red cube next to the blue square"
"Sort all objects by size"
"Arrange objects in a triangle"
"What's the largest object?"
"Place the smallest object in the center"
```

### Interactive Commands

```
tools   - List available tools
clear   - Clear conversation history
switch  - Switch LLM provider
quit    - Exit interactive mode
```

---

## 🎯 Use Cases

### 1. Research & Development
- Rapid prototyping of robot behaviors
- Testing manipulation strategies
- Human-robot interaction studies

### 2. Education
- Teaching robotics concepts
- Demonstrating AI integration
- Student projects

### 3. Industrial Automation
- Pick-and-place tasks
- Quality control sorting
- Assembly line operations

### 4. Warehouse & Logistics
- Object sorting
- Inventory management
- Package handling

### 5. Assistive Robotics
- Object retrieval
- Workspace organization
- Personalized assistance

---

## Getting Help

**Resources:**
- [API Reference & Architecture](api/index.md) - Complete API documentation
- [GitHub Issues](https://github.com/dgaida/robot_mcp/issues) - Report bugs
- [Example Scripts](../examples/) - See working examples
- **MCP Documentation**: https://modelcontextprotocol.io

**Before Opening an Issue:**

- [ ] Redis is running
- [ ] Server started successfully (check logs)
- [ ] At least one API key configured
- [ ] Client can connect (test with curl)
- [ ] Objects are visible to camera
- [ ] Tried examples first

---

## Next Steps

1. ✅ Complete Quick Start setup
2. ✅ Try interactive mode with basic commands
3. ✅ Run example scripts to see capabilities
4. ✅ Explore different LLM providers
5. ✅ Try Gradio web interface
6. ✅ Review API Reference for advanced features
7. ✅ Create your own automation scripts

**Happy robot commanding! 🤖✨**

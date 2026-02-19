# Robot MCP Gradio GUI - Setup Guide

## Quick Start

### 1. Create Anaconda Environment

```bash
# Navigate to robot_mcp directory
cd robot_mcp

# Create environment from YAML file
conda env create -f robot_gui/environment.yml

# Activate environment
conda activate robot_mcp_gui
```

### 2. Configure API Keys

Create or edit `secrets.env` file:

```bash
# At least one LLM API key is required

# OpenAI (GPT-4o, GPT-4o-mini)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# Groq (Free tier available!)
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxx

# Google Gemini
GEMINI_API_KEY=AIzaSy-xxxxxxxxxxxxxxxx

# Optional: ElevenLabs for TTS (if using robot_environment audio)
ELEVENLABS_API_KEY=your_key
```

### 3. Start Required Services

**Terminal 1: Start Redis Server**
```bash
docker run -p 6379:6379 redis:alpine
```

**Terminal 2: Start MCP Server**
```bash
conda activate robot_mcp_gui
python server/fastmcp_robot_server_unified.py --robot niryo --verbose
```

**Terminal 3: Start Object Detection (optional but recommended)**
```bash
# In vision_detect_segment repository
python scripts/detect_objects_publish_annotated_frames.py
```

### 4. Launch Gradio App

```bash
conda activate robot_mcp_gui
python robot_gui/mcp_app.py --api groq
```

## Usage Options

### Basic Usage
```bash
# Default: Groq API, Niryo robot, simulation mode
python robot_gui/mcp_app.py
```

### OpenAI API
```bash
python robot_gui/mcp_app.py --api openai --model gpt-4o-mini
```

### Gemini API
```bash
python robot_gui/mcp_app.py --api gemini --model gemini-2.0-flash-exp
```

### Real Robot (no simulation)
```bash
python robot_gui/mcp_app.py --api groq --no-simulation --robot niryo
```

### Custom Redis Server
```bash
python robot_gui/mcp_app.py --redis-host 192.168.1.100 --redis-port 6380
```

### Public Share Link
```bash
python robot_gui/mcp_app.py --share
```

### Custom Port
```bash
python robot_gui/mcp_app.py --server-port 8080
```

## Full Command-Line Options

```bash
python robot_gui/mcp_app.py --help

Options:
  --api {openai,groq,gemini,ollama}
                        LLM provider (default: groq)
  --model MODEL         Specific model name
  --robot {niryo,widowx}
                        Robot type (default: niryo)
  --no-simulation       Use real robot hardware
  --redis-host HOST     Redis server host (default: localhost)
  --redis-port PORT     Redis server port (default: 6379)
  --share               Create public Gradio link
  --server-port PORT    Gradio server port (default: 7860)
```

## Features

### 1. Live Object Detection Visualization
- Real-time annotated frames from Redis
- Bounding boxes and segmentation masks
- Object labels and confidence scores
- Auto-refresh at ~10 FPS

### 2. Multi-LLM Chat Interface
- Support for OpenAI, Groq, Gemini, Ollama
- Natural language robot control
- Tool call tracking
- Chain-of-thought reasoning display

### 3. Voice Input
- Speech-to-text via Whisper
- Click "🎤 Record Voice" button
- Speak your command
- Automatic transcription to text

### 4. System Status Monitor
- MCP server connection status
- Redis connection status
- Speech-to-text availability
- Current LLM provider and model

### 5. Example Commands
- Pre-filled example tasks
- Click to populate input field
- Covers common robot operations

## Troubleshooting

### "MCP server not connected"
1. Check MCP server is running (Terminal 2)
2. Verify server URL: http://localhost:8000
3. Click "🔌 Connect to MCP Server" button

### "Redis not connected"
1. Check Redis is running: `docker ps`
2. Test connection: `redis-cli ping` (should return "PONG")
3. Verify host/port settings

### "Waiting for frames..."
1. Start object detection pipeline (Terminal 3)
2. Check camera is publishing to Redis
3. Verify stream name: `annotated_camera`

### Speech-to-text not working
1. Check microphone permissions
2. Verify Whisper model downloaded
3. Check GPU/CPU availability for inference

### Slow response times
1. Use faster LLM: Groq (very fast, free tier)
2. Use smaller model: `gpt-4o-mini`, `llama-3.1-8b-instant`
3. Enable GPU for speech-to-text

## Architecture Overview

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Gradio    │  MCP    │              │  Robot  │             │
│     GUI     │◄───────►│  MCP Server  │◄───────►│   Niryo/    │
│  (Browser)  │Protocol │   (FastMCP)  │   API   │   WidowX    │
└─────────────┘         └──────────────┘         └─────────────┘
      ▲                         │                       │
      │                         ▼                       │
      │                   ┌──────────┐                 │
      │                   │  Redis   │                 │
      │                   │ Streams  │◄────────────────┘
      │                   └──────────┘
      │                         │
      └─────────────────────────┘
         (Annotated Frames)
```

## Development Tips

### Enable Verbose Logging
```bash
# MCP Server
python server/fastmcp_robot_server_unified.py --verbose

# Check logs
tail -f log/mcp_server_*.log
```

### Test Redis Connection
```python
from redis_robot_comm import RedisImageStreamer

streamer = RedisImageStreamer()
result = streamer.get_latest_image()
print("Redis OK" if result else "No frames")
```

### Test MCP Connection
```bash
curl http://localhost:8000/sse
# Should return event stream
```

## GPU Acceleration

### Enable CUDA (if available)

Edit `environment.yml`:
```yaml
dependencies:
  - pytorch-cuda=11.8  # Or your CUDA version
  # Remove: - cpuonly
```

Recreate environment:
```bash
conda env remove -n robot_mcp_gui
conda env create -f robot_gui/environment.yml
```

## Support

For issues or questions:
- Check logs in `log/` directory
- See main README.md
- Open issue on GitHub

## License

MIT License - see LICENSE file for details.

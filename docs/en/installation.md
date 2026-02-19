# Installation Guide - Robot MCP v0.3.0

Complete installation guide for the Robot MCP Control System with multi-LLM support.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4 GB (8 GB recommended)
- **GPU**: Optional, but recommended for better performance
- **Redis**: Required for camera streaming and communication

### Software Prerequisites

```bash
# Check Python version
python --version  # or python3 --version

# Update pip
pip install --upgrade pip

# Git (for development)
git --version
```

## 🚀 Installation

### Option 1: Standard Installation (Recommended)

```bash
# 1. Clone repository (if not already done)
git clone https://github.com/dgaida/robot_mcp.git
cd robot_mcp

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Install package with all dependencies
pip install -e ".[all]"
```

### Option 2: Minimal Installation

Only MCP Server and Client, without GUI:

```bash
pip install -e "."
```

### Option 3: Custom Installation

Choose only the components you need:

```bash
# Only GUI components
pip install -e ".[gui]"

# Development tools
pip install -e ".[dev]"

# Documentation tools
pip install -e ".[docs]"

# Everything
pip install -e ".[all]"
```

## 📦 Dependencies by Component

### Core (Always Required)

These are installed automatically with the base package:

```bash
fastmcp>=0.1.0              # Modern MCP implementation
llm_client                  # Multi-LLM support (from GitHub)
python-dotenv>=1.0.0        # Environment variable management
robot-environment           # Robot control (from GitHub)
text2speech                 # TTS integration (from GitHub)
speech2text                 # STT integration (from GitHub)
```

### GUI Components

```bash
pip install -e ".[gui]"

# Installs:
# - gradio>=4.0.0
# - redis_robot_comm (from GitHub)
```

### Development Tools

```bash
pip install -e ".[dev]"

# Installs:
# - pytest>=7.0.0
# - pytest-asyncio>=0.21.0
# - pytest-cov>=4.0.0
# - black>=23.0.0
# - ruff>=0.1.0
# - mypy>=1.0.0
# - isort>=5.12.0
```

### Documentation

```bash
pip install -e ".[docs]"

# Installs:
# - sphinx>=5.0.0
# - sphinx-rtd-theme>=1.2.0
# - myst-parser>=1.0.0
```

## 🔑 API Keys Configuration

The system now supports **4 LLM providers**. You need **at least one API key** (or use Ollama for local/offline operation).

### Supported Providers

| Provider | Cost | Speed | Best For |
|----------|------|-------|----------|
| **OpenAI** | $$ | Fast | Production, complex reasoning |
| **Groq** | Free tier | Very Fast | Development, prototyping |
| **Gemini** | Free tier | Fast | Long context, multimodal |
| **Ollama** | Free (local) | Variable | Offline, privacy |

### 1. Get API Keys

**OpenAI** (GPT-4o, GPT-4o-mini):
```bash
# 1. Go to https://platform.openai.com/api-keys
# 2. Create account or sign in
# 3. Click "Create new secret key"
# 4. Copy the key (starts with sk-...)
```

**Groq** (Kimi, Llama, Mixtral) - **Free tier available**:
```bash
# 1. Go to https://console.groq.com/keys
# 2. Create account or sign in
# 3. Click "Create API Key"
# 4. Copy the key (starts with gsk_...)
```

**Google Gemini** (Gemini 2.0, 2.5):
```bash
# 1. Go to https://aistudio.google.com/apikey
# 2. Sign in with Google account
# 3. Click "Create API Key"
# 4. Copy the key (starts with AIzaSy...)
```

**Ollama** (Local models) - **No API key needed**:
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a model (e.g., llama3.2:1b)
ollama pull llama3.2:1b

# 3. Start Ollama service
ollama serve
```

### 2. Optional: ElevenLabs API Key

For enhanced text-to-speech quality:

```bash
# 1. Go to https://elevenlabs.io
# 2. Create account
# 3. Go to Settings > API Keys
# 4. Copy the key
```

### 3. Store Keys in secrets.env

```bash
# Copy template
cp secrets.env.template secrets.env

# Edit the file
nano secrets.env  # or use your preferred editor
```

**secrets.env content:**

```bash
# ============================================================================
# LLM Provider API Keys (add at least one)
# ============================================================================

# OpenAI (best for production)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Groq (fastest, free tier available) - RECOMMENDED FOR GETTING STARTED
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google Gemini (free tier available)
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Ollama - No key needed, just install and run
# Installation: curl -fsSL https://ollama.ai/install.sh | sh

# ============================================================================
# Optional: Enhanced TTS
# ============================================================================

# ElevenLabs (for better voice quality)
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Set file permissions (Linux/Mac):**

```bash
chmod 600 secrets.env
```

**Add to .gitignore:**

```bash
echo "secrets.env" >> .gitignore
```

### 4. API Selection Priority

If you configure multiple API keys, the system uses this priority:

1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **Groq** (if `GROQ_API_KEY` is set)
3. **Gemini** (if `GEMINI_API_KEY` is set)
4. **Ollama** (fallback if installed, no key needed)

**Override with --api flag:**

```bash
# Force specific provider
python client/fastmcp_universal_client.py --api groq
python client/fastmcp_universal_client.py --api gemini
python client/fastmcp_universal_client.py --api ollama
```

## 🔧 Verification of Installation

### 1. Test Python Imports

```python
# test_imports.py
import fastmcp
from llm_client import LLMClient
from robot_environment import Environment
from text2speech import Text2Speech
import gradio

print("✓ All core imports successful!")
```

```bash
python test_imports.py
```

### 2. Check Package Installation

```bash
# Show installed package
pip list | grep robot-mcp

# Show package details
pip show robot-mcp
```

Expected output:
```
Name: robot-mcp
Version: 0.3.0
Summary: Natural language robot control using FastMCP with multi-LLM support
```

### 3. Test Command-line Scripts

The package installs several command-line scripts:

```bash
# Test if scripts are installed
robot-fastmcp-server --help
robot-universal-client --help
robot-groq-client --help
robot-gui --help
robot-examples --help
```

### 4. Test MCP Server

```bash
# Start server in background
python server/fastmcp_robot_server.py --robot niryo &

# Wait for startup
sleep 5

# Test if server is running
curl http://127.0.0.1:8000/sse

# Expected: Connection established or SSE stream starts

# Stop server
pkill -f fastmcp_robot_server
```

### 5. Test Universal Client (Multi-LLM)

```bash
# Auto-detect available API
python client/fastmcp_universal_client.py --command "What LLM provider am I using?"

# Test specific provider
python client/fastmcp_universal_client.py --api groq --command "Hello"
python client/fastmcp_universal_client.py --api openai --command "Hello"
```

### 6. Verify Redis Connection

```bash
# Check if Redis is running
redis-cli ping

# Expected output: PONG

# If Redis is not running:
docker run -p 6379:6379 redis:alpine
# or
redis-server
```

## 🛠️ Common Installation Problems

### Problem: Package not found after installation

**Symptom:**
```
ModuleNotFoundError: No module named 'robot_mcp'
```

**Solution:**
```bash
# Make sure you're in the virtual environment
which python  # Should show venv path

# Reinstall package
pip install -e ".[all]"

# Verify installation
pip show robot-mcp
```

---

### Problem: llm_client installation failed

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement llm_client
```

**Solution:**
```bash
# Install directly from GitHub
pip install git+https://github.com/dgaida/llm_client.git

# Then reinstall robot-mcp
pip install -e ".[all]"
```

---

### Problem: robot-environment installation failed

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement robot-environment
```

**Solution:**
```bash
# Install directly from GitHub
pip install git+https://github.com/dgaida/robot_environment.git

# Then reinstall robot-mcp
pip install -e ".[all]"
```

---

### Problem: OpenAI client not available

**Symptom:**
```
ImportError: No module named 'openai'
```

**Solution:**
```bash
# The llm_client package should install this, but if not:
pip install openai

# Or reinstall with all dependencies
pip install -e ".[all]" --force-reinstall
```

---

### Problem: Groq client not available

**Symptom:**
```
ImportError: No module named 'groq'
```

**Solution:**
```bash
# Install Groq client
pip install groq

# Or reinstall with all dependencies
pip install -e ".[all]" --force-reinstall
```

---

### Problem: Ollama not responding

**Symptom:**
```
Error: Ollama server not responding
```

**Solution:**
```bash
# Check if Ollama is installed
ollama --version

# If not installed:
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model
ollama pull llama3.2:1b

# Test
ollama run llama3.2:1b "Hello"
```

---

### Problem: fastmcp not found

**Symptom:**
```
ModuleNotFoundError: No module named 'fastmcp'
```

**Solution:**
```bash
# Install fastmcp
pip install fastmcp

# Or reinstall robot-mcp with all dependencies
pip install -e ".[all]" --force-reinstall
```

---

### Problem: gradio installation failed

**Symptom:**
```
ERROR: Could not install packages due to an OSError
```

**Solution:**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install gradio separately
pip install gradio

# Then install GUI components
pip install -e ".[gui]"
```

---

### Problem: Redis connection error

**Symptom:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution:**
```bash
# Start Redis with Docker
docker run -p 6379:6379 redis:alpine

# Or install and start Redis locally
# Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis

# macOS:
brew install redis
brew services start redis

# Test connection
redis-cli ping
# Should return: PONG
```

---

### Problem: API key not recognized

**Symptom:**
```
Error: No valid API key found
```

**Solution:**
```bash
# Check if secrets.env exists
ls -la secrets.env

# Check file content (without revealing keys)
cat secrets.env | grep -o ".*_API_KEY=.*" | sed 's/=.*/=***/'

# Make sure the file is in the project root
pwd  # Should show robot_mcp directory

# Check environment variables are loaded
python -c "from dotenv import load_dotenv; load_dotenv('secrets.env'); import os; print('GROQ' in os.environ)"
# Should print: True (if Groq key is set)
```

---

## 💻 Platform-Specific Notes

### Windows

```powershell
# Use PowerShell
# Activate virtual environment:
venv\Scripts\Activate.ps1

# If execution policy blocks:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again
venv\Scripts\Activate.ps1

# Install package
pip install -e ".[all]"
```

### macOS (M1/M2 Apple Silicon)

```bash
# ARM64 architecture
# Use miniforge for better compatibility
brew install miniforge
conda init zsh  # or bash

# Create new environment
conda create -n robot-mcp python=3.10
conda activate robot-mcp

# Install dependencies
pip install -e ".[all]"

# Note: Some packages may need ARM64 builds
# If issues occur, use Rosetta mode:
arch -x86_64 pip install -e ".[all]"
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libopencv-dev \
    build-essential \
    redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Normal installation
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

## 📄 Update & Uninstallation

### Update Package

```bash
# For development version (editable install)
cd robot_mcp
git pull origin master
pip install -e ".[all]" --upgrade

# Update only dependencies
pip install -e ".[all]" --upgrade --force-reinstall --no-cache-dir
```

### Uninstall Package

```bash
# Remove package
pip uninstall robot-mcp

# Remove virtual environment
deactivate
rm -rf venv/

# Remove all cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
```

## 🧪 Testing the Installation

### Run Test Suite

```bash
# All tests
pytest

# Only unit tests
pytest -m unit

# With coverage report
pytest --cov=client --cov=server --cov=robot_gui

# Verbose output
pytest -v -s
```

### Quick Functionality Test

```bash
# 1. Start Redis
docker run -p 6379:6379 redis:alpine &

# 2. Start MCP Server
python server/fastmcp_robot_server.py --robot niryo &
sleep 5

# 3. Test universal client with auto-detection
python client/fastmcp_universal_client.py --command "test connection"

# 4. Test specific provider
python client/fastmcp_universal_client.py --api groq --command "What API am I using?"

# 5. Cleanup
pkill -f fastmcp_robot_server
pkill -f redis-server
```

## 📚 Next Steps

After successful installation:

1. **Configure API Keys**: Complete `secrets.env` with at least one API key
2. **Quick Start**: See `docs/mcp_setup_guide.md` for usage examples
3. **Examples**: Try `python examples/universal_examples.py workspace_scan`
4. **GUI**: Launch `python robot_gui/mcp_app.py --robot niryo`
5. **Documentation**: Read `README.md` for system overview

## 🎯 Recommended Setup for Beginners

**Fastest way to get started:**

```bash
# 1. Get a free Groq API key (fastest, no credit card)
# Visit: https://console.groq.com/keys

# 2. Install with all components
pip install -e ".[all]"

# 3. Add Groq key to secrets.env
echo "GROQ_API_KEY=gsk_your_key_here" > secrets.env

# 4. Start Redis
docker run -p 6379:6379 redis:alpine &

# 5. Start server
python server/fastmcp_robot_server.py --robot niryo &

# 6. Try the universal client (will auto-detect Groq)
python client/fastmcp_universal_client.py

# 7. Type: "What objects do you see?"
```

## 🆘 Getting Help

If problems persist:

1. **Check Logs**: `tail -f log/mcp_server_*.log`
2. **GitHub Issues**: https://github.com/dgaida/robot-mcp/issues
3. **Documentation**: See `docs/` directory for detailed guides
4. **Community**: Ask in GitHub Discussions

## 📝 Installation Checklist

Use this checklist to verify complete installation:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Package installed: `pip show robot-mcp` works
- [ ] At least one API key configured in `secrets.env`
- [ ] Redis server running: `redis-cli ping` returns PONG
- [ ] MCP server starts: `python server/fastmcp_robot_server.py`
- [ ] Universal client works: `python client/fastmcp_universal_client.py --command "test"`
- [ ] LLM provider detected: Check client output shows API name
- [ ] Command-line scripts available: `robot-universal-client --help`

## 🔍 Troubleshooting Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| ModuleNotFoundError | `pip install -e ".[all]" --force-reinstall` |
| Redis connection error | `docker run -p 6379:6379 redis:alpine` |
| No API key found | Check `secrets.env` exists and has valid keys |
| Server won't start | Check port 8000 is free: `lsof -i :8000` |
| Ollama not responding | `ollama serve` then `ollama pull llama3.2:1b` |
| Import errors | Ensure virtual environment is activated |
| Permission denied | `chmod 600 secrets.env` |

---

**Installation complete? → Start with `docs/mcp_setup_guide.md`! 🚀**

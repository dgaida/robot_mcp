# MCP Robot Control System

Natural language robot control using Model Context Protocol (MCP) and Groq's LLM API.

## 🎯 Quick Start (60 seconds)

```bash
# 1. Clone and navigate
cd robot_mcp

# 2. Run quick start script
chmod +x quickstart_mcp.sh
./quickstart_mcp.sh

# 3. Follow the interactive menu
# Choose option 3: "Run MCP client (interactive chat)"
```

## 📋 What's Included

### Files Overview

```
robot_mcp/
├── mcp_robot_server.py         # MCP server exposing robot tools
├── mcp_groq_client.py          # Groq-powered MCP client
├── mcp_server_launcher.py      # Utility launcher script
├── examples_mcp_client.py      # Example scripts collection
├── requirements.txt            # Python dependencies
├── mcp_quickstart.sh           # Quick setup script
└── README.md                   # This file
```

### Components

1. **MCP Server** (`mcp_robot_server.py`)
   - Exposes 12 robot control tools via MCP
   - Runs as stdio server for MCP clients
   - Works with Claude Desktop or custom clients

2. **Groq Client** (`mcp_groq_client.py`)
   - Interactive chat interface
   - Uses Groq's fast LLM inference
   - Automatic tool calling and execution
   - Conversation history tracking

3. **Launcher** (`mcp_server_launcher.py`)
   - Unified command-line interface
   - Test, run, configure, and inspect
   - Supports both robots and simulation

4. **Examples** (`examples_mcp_client.py`)
   - 15 ready-to-run examples
   - From simple scans to complex tasks
   - Learning resource for new users

## 🚀 Installation

### Option 1: Quick Start Script (Recommended)

```bash
./mcp_quickstart.sh
```

The script will:
- Check dependencies
- Create virtual environment
- Install packages
- Prompt for Groq API key
- Present interactive menu

### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install mcp groq
pip install -e ".[all]"

# Set API key
export GROQ_API_KEY="your_groq_api_key"
```

### Get Groq API Key

1. Visit https://console.groq.com/keys
2. Sign up or log in
3. Create new API key
4. Copy and save securely

## 💻 Usage

### 1. Interactive Chat Mode

Talk to your robot in natural language:

```bash
python mcp_groq_client.py --robot niryo

# With simulation
python mcp_groq_client.py --robot niryo --simulation
```

**Example Conversation:**

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

### 2. Single Command Mode

Execute one command:

```bash
python mcp_groq_client.py \
  --command "Pick up the pencil and place it at [0.2, 0.1]"
```

### 3. Run Examples

Try pre-built examples:

```bash
# List all examples
python examples_mcp_client.py --help

# Run workspace scan
python examples_mcp_client.py workspace_scan

# Run with simulation
python examples_mcp_client.py sort_by_size --simulation

# Run all examples
python examples_mcp_client.py all
```

### 4. Use with Claude Desktop

Generate configuration:

```bash
python mcp_launcher.py config --output claude_config.json
```

Copy content to Claude Desktop config:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Restart Claude Desktop and the tools will be available!

### 5. Test Connection

Before running examples, test your setup:

```bash
python mcp_launcher.py test --robot niryo
```

## 🛠️ Available Tools

The MCP server exposes these tools to LLMs:

### Robot Control
- `pick_place_object` - Pick and place in one operation
- `pick_object` - Pick up an object
- `place_object` - Place a held object
- `push_object` - Push objects (for unpickable items)
- `move_to_observation_pose` - Position for observation

### Object Detection
- `get_detected_objects` - List all objects
- `get_object_at_location` - Find object at coordinates
- `get_nearest_object` - Find closest object
- `get_largest_object` - Get biggest object
- `get_smallest_object` - Get smallest object

### Information
- `get_workspace_info` - Workspace dimensions
- `speak` - Text-to-speech output

## 📚 Example Tasks

### Simple Tasks

```python
# Workspace scan
"What objects do you see?"

# Basic pick and place
"Pick up the pencil and place it at [0.2, 0.1]"

# Relative placement
"Move the red cube to the left of the blue square"

# Find objects
"What's the largest object?"
"What's near position [0.15, 0.0]?"
```

### Intermediate Tasks

```python
# Sorting
"Sort all objects by size from smallest to largest"

# Pattern creation
"Arrange objects in a triangle"

# Conditional logic
"If there's a pencil, move it to [0.2, 0]. Otherwise, tell me what's available"

# Grouping
"Group objects by color: red on left, blue on right"
```

### Advanced Tasks

```python
# Multi-step operations
"Execute: 1) Find all objects 2) Move smallest to [0.15, 0.1] 3) Move largest right of smallest 4) Report positions"

# Complex reasoning
"Organize the workspace: cubes on left, cylinders in middle, everything else on right, aligned in rows"

# Error recovery
"Try to pick up 'nonexistent'. If that fails, pick up any object and place it at [0.2, 0.0]"
```

## 🎮 Command Patterns

### Object Queries
```
"What objects..."
"Show me all..."
"Find the..."
"Which object is..."
"How many..."
```

### Pick and Place
```
"Pick up the [object] and place it..."
"Move [object] to..."
"Put [object] next to/above/below..."
"Swap positions of..."
```

### Spatial Reasoning
```
"Arrange in a [pattern]"
"Sort by [criteria]"
"Group by [attribute]"
"Place at [coordinates]"
"Stack [objects]"
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
GROQ_API_KEY=gsk_your_api_key_here
ROBOT_ID=niryo
USE_SIMULATION=false
GROQ_MODEL=llama-3.3-70b-versatile
```

Load with:
```bash
export $(cat .env | xargs)
```

### Groq Models

Choose based on your needs:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| llama-3.3-70b-versatile | Fast | Excellent | General use (default) |
| llama-3.1-8b-instant | Very Fast | Good | Simple commands |
| mixtral-8x7b-32768 | Fast | Very Good | Complex reasoning |

Change model:
```bash
python mcp_groq_client.py --model llama-3.1-8b-instant
```

### Server Configuration

Customize server behavior in `mcp_robot_server.py`:

```python
# Modify system prompt
self.system_prompt = """Your custom instructions..."""

# Add custom tools
@self.server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # Add your custom tools here
    ]
```

## 🔧 Troubleshooting

### Common Issues

#### "Connection refused"

**Cause**: MCP server not running or path incorrect

**Solution**:
```bash
# Test server manually
python mcp_robot_server.py niryo false

# Check server path in client
python mcp_groq_client.py --server /full/path/to/mcp_robot_server.py
```

#### "Invalid API key"

**Cause**: Groq API key not set or incorrect

**Solution**:
```bash
# Check if set
echo $GROQ_API_KEY

# Set temporarily
export GROQ_API_KEY="gsk_your_key"

# Or pass directly
python mcp_groq_client.py --api-key "gsk_your_key"
```

#### "No objects detected"

**Cause**: Robot not in observation pose or detection not initialized

**Solution**:
```bash
# First command should always be:
"Move to observation pose"
# Then wait 2-3 seconds before querying objects
```

#### Robot doesn't move

**Cause**: Simulation flag mismatch or robot not connected

**Solution**:
```bash
# Check robot connection
python mcp_server_launcher.py test --robot niryo

# Verify simulation flag matches your setup
python mcp_groq_client.py --robot niryo --simulation  # For sim
python mcp_groq_client.py --robot niryo               # For real
```

#### Slow responses

**Cause**: Large conversation history or slow model

**Solutions**:
```bash
# In interactive mode, type:
clear

# Use faster model
python mcp_groq_client.py --model llama-3.1-8b-instant

# Single command mode (no history)
python mcp_groq_client.py --command "your command"
```

### Debug Mode

Enable verbose output:

```bash
# Server side
python mcp_robot_server.py niryo false  # verbose=True in code

# Check logs
tail -f robot_mcp_*.log
```

## 📖 Advanced Usage

### Custom Scripts

Create your own automation:

```python
import asyncio
from client.mcp_groq_client import RobotMCPClient

async def my_task():
    client = RobotMCPClient(
        groq_api_key="your_key",
        robot_id="niryo",
        use_simulation=False
    )
    
    await client.connect()
    
    # Your commands
    await client.chat("What objects do you see?")
    await client.chat("Sort them by size")
    await client.chat("Create a triangle pattern")
    
    await client.disconnect()

asyncio.run(my_task())
```

### Batch Processing

Process multiple commands from a file:

```python
# commands.txt
What objects do you see?
Move the largest object to [0.2, 0.0]
Arrange remaining objects in a line
Report final positions
```

```python
import asyncio
from client.mcp_groq_client import RobotMCPClient

async def batch_process(commands_file):
    client = RobotMCPClient(groq_api_key="your_key")
    await client.connect()
    
    with open(commands_file) as f:
        for line in f:
            command = line.strip()
            if command and not command.startswith('#'):
                print(f"Executing: {command}")
                await client.chat(command)
                await asyncio.sleep(2)
    
    await client.disconnect()

asyncio.run(batch_process('commands.txt'))
```

### Integration with Other Systems

```python
# Example: Web API endpoint
from fastapi import FastAPI
from client.mcp_groq_client import RobotMCPClient

app = FastAPI()
client = None

@app.on_event("startup")
async def startup():
    global client
    client = RobotMCPClient(groq_api_key="your_key")
    await client.connect()

@app.post("/robot/command")
async def execute_command(command: str):
    response = await client.chat(command)
    return {"response": response}

@app.on_event("shutdown")
async def shutdown():
    await client.disconnect()
```

### Custom Tool Development

Add your own tools to the MCP server:

```python
# In mcp_robot_server.py

# 1. Add to list_tools()
Tool(
    name="custom_calibration",
    description="Perform custom calibration routine",
    inputSchema={
        "type": "object",
        "properties": {
            "calibration_type": {
                "type": "string",
                "enum": ["full", "quick", "visual"]
            }
        },
        "required": ["calibration_type"]
    }
)

# 2. Add handler in call_tool()
elif name == "custom_calibration":
    cal_type = arguments["calibration_type"]
    # Your calibration logic
    result = perform_calibration(cal_type)
    return [TextContent(
        type="text",
        text=f"Calibration complete: {result}"
    )]
```

## 🔐 Security

### API Key Security

```bash
# Never commit API keys to git
echo "GROQ_API_KEY=*" >> .gitignore
echo ".env" >> .gitignore

# Use environment variables
export GROQ_API_KEY="your_key"

# Or use a secrets manager
# Example with AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id groq-api-key
```

### Workspace Safety

Add boundary checking:

```python
# In mcp_robot_server.py
def validate_coordinates(x, y):
    """Ensure coordinates are within safe bounds."""
    if not (0.05 <= x <= 0.35 and -0.20 <= y <= 0.20):
        raise ValueError(f"Coordinates [{x}, {y}] outside safe bounds")
```

### Access Control

Implement rate limiting and logging:

```python
import time
from collections import defaultdict

class RobotMCPServer:
    def __init__(self, ...):
        self.request_counts = defaultdict(int)
        self.request_times = defaultdict(list)
    
    def check_rate_limit(self, user_id, max_requests=10, time_window=60):
        """Rate limit: max_requests per time_window seconds."""
        now = time.time()
        times = self.request_times[user_id]
        
        # Remove old entries
        times = [t for t in times if now - t < time_window]
        
        if len(times) >= max_requests:
            raise Exception("Rate limit exceeded")
        
        times.append(now)
        self.request_times[user_id] = times
```

## 📊 Monitoring and Logging

### Operation Logging

Track all robot operations:

```python
import logging
import json
from datetime import datetime

logging.basicConfig(
    filename=f'robot_ops_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_operation(tool_name, arguments, result, duration):
    """Log each operation."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
        "duration_ms": duration * 1000
    }
    logging.info(json.dumps(log_entry))
```

### See contents in Log File in Real-Time in Windows

Open PowerShell and change to folder of robot_mcp package (cd).

```bash
Get-Content -Path mcp_server_*.log -Wait -Tail 0
```

### Performance Metrics

Track system performance:

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_duration": 0,
            "operations_by_type": {}
        }
    
    def record_operation(self, tool_name, success, duration):
        self.metrics["total_operations"] += 1
        if success:
            self.metrics["successful_operations"] += 1
        else:
            self.metrics["failed_operations"] += 1
        self.metrics["total_duration"] += duration
        
        if tool_name not in self.metrics["operations_by_type"]:
            self.metrics["operations_by_type"][tool_name] = 0
        self.metrics["operations_by_type"][tool_name] += 1
    
    def get_report(self):
        return {
            "success_rate": self.metrics["successful_operations"] / 
                          max(1, self.metrics["total_operations"]),
            "avg_duration": self.metrics["total_duration"] / 
                          max(1, self.metrics["total_operations"]),
            "most_used_tool": max(
                self.metrics["operations_by_type"].items(),
                key=lambda x: x[1],
                default=("none", 0)
            )[0]
        }
```

## 🧪 Testing

### Unit Tests

```python
# test_mcp_client.py
import pytest
from mcp_groq_client import RobotMCPClient

@pytest.mark.asyncio
async def test_connection():
    client = RobotMCPClient(
        groq_api_key="test_key",
        use_simulation=True
    )
    await client.connect()
    assert len(client.available_tools) > 0
    await client.disconnect()

@pytest.mark.asyncio
async def test_object_detection():
    client = RobotMCPClient(groq_api_key="test_key")
    await client.connect()
    response = await client.chat("What objects do you see?")
    assert "object" in response.lower()
    await client.disconnect()
```

Run tests:
```bash
pytest test_mcp_client.py -v
```

### Integration Tests

Test full workflows:

```python
# test_integration.py
import asyncio
from client.mcp_groq_client import RobotMCPClient

async def test_full_workflow():
    """Test complete pick and place workflow."""
    client = RobotMCPClient(groq_api_key="your_key", use_simulation=True)
    await client.connect()
    
    # Step 1: Detect objects
    response1 = await client.chat("What objects do you see?")
    assert "object" in response1.lower()
    
    # Step 2: Pick and place
    response2 = await client.chat("Pick up the first object and place it at [0.2, 0.1]")
    assert "success" in response2.lower() or "done" in response2.lower()
    
    # Step 3: Verify
    response3 = await client.chat("What's at position [0.2, 0.1]?")
    assert "object" in response3.lower()
    
    await client.disconnect()
    print("✓ Full workflow test passed")

if __name__ == "__main__":
    asyncio.run(test_full_workflow())
```

## 🚀 Deployment

### Production Deployment

For production use:

1. **Use Process Manager**
```bash
# Install supervisor
sudo apt-get install supervisor

# Create config: /etc/supervisor/conf.d/mcp_server.conf
[program:mcp_server]
command=/path/to/venv/bin/python /path/to/mcp_robot_server.py niryo false
directory=/path/to/robot-environment
user=robotuser
autostart=true
autorestart=true
stderr_logfile=/var/log/mcp_server.err.log
stdout_logfile=/var/log/mcp_server.out.log
```

2. **Use Docker**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "mcp_robot_server.py", "niryo", "false"]
```

3. **Environment Configuration**
```bash
# Use systemd for production
sudo systemctl enable mcp-server
sudo systemctl start mcp-server
```

## 📝 Best Practices

### Command Design

✅ **Good Commands:**
- "Pick up the red cube and place it at [0.2, 0.1]"
- "Find the largest object and move it to the center"
- "Sort all objects by size, smallest to largest"

❌ **Avoid:**
- "Do something with that"
- "Move it over there"
- "Fix the workspace"

### Error Handling

Always provide fallback options:

```python
"If there's a pencil, move it to [0.2, 0]. 
Otherwise, move any available object."
```

### Performance

- Clear conversation history periodically
- Use single command mode for scripts
- Choose appropriate Groq model for task complexity

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## 📄 License

MIT License

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` folder
- **Examples**: Run `python examples_mcp_client.py --help`

## 🎓 Learning Resources

- [MCP Documentation](https://modelcontextprotocol.io)
- [Groq API Docs](https://console.groq.com/docs)
- [Robot Environment README](README.md)

---

Made with ❤️ for robotic automation

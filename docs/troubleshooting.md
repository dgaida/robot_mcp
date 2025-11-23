# Robot MCP - Troubleshooting

Common issues and solutions for the Robot MCP system.

## Table of Contents

- [Connection Issues](#connection-issues)
- [Object Detection Problems](#object-detection-problems)
- [Robot Movement Issues](#robot-movement-issues)
- [LLM and Tool Call Problems](#llm-and-tool-call-problems)
- [Performance Issues](#performance-issues)
- [Hardware Problems](#hardware-problems)
- [Development and Testing](#development-and-testing)

---

## Connection Issues

### FastMCP Server Won't Start

**Symptoms:**
- Server process exits immediately
- Connection timeout errors
- "Port already in use" messages

**Solutions:**

1. **Check if port 8000 is already in use:**
```bash
# Linux/Mac
lsof -i :8000

# Windows
netstat -ano | findstr :8000

# Kill existing process if found
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

2. **Check Python dependencies:**
```bash
pip install --upgrade fastmcp groq robot-environment
```

3. **Start server manually with verbose logging:**
```bash
python server/fastmcp_robot_server.py --robot niryo --verbose
```

4. **Check log files:**
```bash
cat log/mcp_server_*.log
```

**Common Error Messages:**

```
ImportError: No module named 'fastmcp'
→ Solution: pip install fastmcp

ModuleNotFoundError: No module named 'robot_environment'
→ Solution: pip install git+https://github.com/dgaida/robot_environment.git

redis.exceptions.ConnectionError
→ Solution: Start Redis: docker run -p 6379:6379 redis:alpine
```

---

### Client Can't Connect to Server

**Symptoms:**
- "Connection refused" errors
- Timeout after 30 seconds
- No response from server

**Solutions:**

1. **Verify server is running:**
```bash
# Check if server is listening
curl http://127.0.0.1:8000/sse

# Should return: Connection established or SSE stream
```

2. **Check firewall settings:**
```bash
# Linux: Allow port 8000
sudo ufw allow 8000

# Windows: Add firewall rule for Python
```

3. **Verify server/client on same network:**
```python
# In client code, ensure correct URL
transport = SSETransport("http://127.0.0.1:8000/sse")
# Not: http://localhost:8000/sse (can cause issues)
```

4. **Test with simple HTTP request:**
```python
import httpx
asyncio.run(httpx.get("http://127.0.0.1:8000/sse"))
```

---

### Groq API Key Issues

**Symptoms:**
- "Invalid API key" errors
- Authentication failures
- Empty responses from LLM

**Solutions:**

1. **Verify API key is set:**
```bash
# Check secrets.env
cat secrets.env | grep GROQ_API_KEY

# Should show: GROQ_API_KEY=gsk_...
```

2. **Test API key:**
```python
from groq import Groq
client = Groq(api_key="your_key")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "test"}]
)
print(response.choices[0].message.content)
```

3. **Regenerate API key:**
- Visit https://console.groq.com/keys
- Create new key
- Update `secrets.env`

4. **Check rate limits:**
```
Error: Rate limit exceeded
→ Solution: Wait or upgrade Groq plan
→ Free tier: 30 requests/minute
```

---

## Object Detection Problems

See [troubleshooting.md](https://github.com/dgaida/robot_environment/docs/troubleshooting.md).

---

## Robot Movement Issues

See [troubleshooting.md](https://github.com/dgaida/robot_environment/docs/troubleshooting.md#robot-movement-issues).

---

## LLM and Tool Call Problems

### LLM Not Calling Tools

**Symptoms:**
- LLM responds in text only
- No robot actions performed
- Tools ignored

**Solutions:**

1. **Verify tools are registered:**
```python
client = RobotFastMCPClient(groq_api_key="key")
await client.connect()

tools = client.available_tools
print(f"Available tools: {[t.name for t in tools]}")
# Should list: pick_place_object, get_detected_objects, etc.
```

2. **Check tool call format:**
```python
# Groq expects specific format
tools_groq = client._convert_tools_to_groq_format()
print(json.dumps(tools_groq[0], indent=2))

# Should have: type, function.name, function.description, function.parameters
```

3. **Improve prompts:**
```python
# ✅ Good: Specific command
"Pick up the pencil at [0.15, -0.05] and place it at [0.2, 0.1]"

# ❌ Bad: Vague request
"Do something with the pencil"
```

4. **Check model capabilities:**
```python
# Some models better at tool calling
# Recommended:
model="moonshotai/kimi-k2-instruct-0905"  # Excellent
model="llama-3.3-70b-versatile"          # Very good

# Less recommended:
model="llama-3.1-8b-instant"  # May miss some tool calls
```

---

### Tool Calls Failing

**Symptoms:**
- "Error executing tool" messages
- Partial execution
- Tool returns errors

**Solutions:**

1. **Check parameter types:**
```python
# ✅ Good: Correct types
pick_place_object(
    object_name="pencil",        # str
    pick_coordinate=[0.15, -0.05],  # List[float]
    place_coordinate=[0.2, 0.1],    # List[float]
    location="right next to"        # str or None
)

# ❌ Bad: Wrong types
pick_place_object(
    object_name=123,  # Should be string
    pick_coordinate="[0.15, -0.05]",  # Should be list
    ...
)
```

2. **Validate coordinates:**
```python
def is_valid_coordinate(coord):
    x, y = coord
    return (0.163 <= x <= 0.337 and
            -0.087 <= y <= 0.087)

if not is_valid_coordinate([x, y]):
    print("Coordinate out of workspace bounds!")
```

3. **Check object exists:**
```python
# Before pick_place_object
obj = get_detected_object([x, y], label="target")
if obj is None:
    print("Object not found at specified location")
    return
```

4. **Review server logs:**
```bash
tail -f log/mcp_server_*.log

# Look for:
# - Tool execution errors
# - Parameter validation failures
# - Robot controller errors
```

---

### Max Iterations Reached

**Symptoms:**
- "Maximum iterations reached" message
- Task incomplete
- Loop detected

**Solutions:**

1. **Increase iteration limit:**
```python
# In client chat() method
max_iterations = 10  # Increase from 4

# For complex tasks, may need more
```

2. **Break down complex tasks:**
```python
# ✅ Good: Step-by-step
"First, find all objects"
# Wait for response
"Now, move the pencil to [0.2, 0.1]"
# Wait for response
"Finally, move the cube next to it"

# ❌ Bad: Everything at once
"Find objects, sort by size, create triangle pattern, report positions"
```

3. **Provide intermediate coordinates:**
```python
# Help LLM with explicit coordinates
"Move pencil from [0.15, -0.05] to [0.2, 0.1], then move cube from [0.20, 0.10] to [0.25, 0.0]"
```

---

## Performance Issues

### Slow Response Times

**Symptoms:**
- Long delays between commands
- System feels sluggish
- Timeouts

**Solutions:**

1. **Use faster Groq model:**
```python
client = RobotFastMCPClient(
    groq_api_key="key",
    model="llama-3.1-8b-instant"  # Fastest
)
```

2. **Optimize object detection:**
```python
# Use YOLO-World instead of OwlV2
objdetect_model_id="yoloworld"

# Reduce detection frequency
time.sleep(1.0)  # In camera loop
```

3. **Enable GPU acceleration:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
```

4. **Reduce conversation history:**
```python
# Limit history size
max_history = 10
client.conversation_history = client.conversation_history[-max_history:]
```

---

### High Memory Usage

**Symptoms:**
- System running out of memory
- Crashes during operation
- Slow performance

**Solutions:**

1. **Use float16 instead of float32:**
```python
Speech2Text(
    device="cuda",
    torch_dtype=torch.float16  # Half precision
)
```

2. **Clear detection cache:**
```python
# Periodically clear old detections
env.clear_detection_cache()
```

3. **Limit image resolution:**
```python
# In camera capture
img = cv2.resize(img, (640, 480))  # Standard resolution
```

4. **Close unused resources:**
```python
# After task completion
await client.disconnect()
env.cleanup()
```

### Rate Limit Errors

**Problem**: Too many requests to Groq

**Solutions**:
1. Add delays between commands
2. Use a less powerful model (llama-3.1-8b-instant)
3. Upgrade Groq plan for higher limits
4. Clear conversation history: type `clear`

---

## Hardware Problems

See [troubleshooting.md](https://github.com/dgaida/robot_environment/docs/troubleshooting.md).

---

## Development and Testing

### Testing Without Robot

**Use Simulation Mode:**
```bash
# Start server in simulation
python server/fastmcp_robot_server.py --robot niryo
# (without --no-simulation flag)
```

**Mock Objects for Testing:**
```python
# Create test objects
test_objects = [
    {"label": "pencil", "x": 0.15, "y": -0.05, "width_m": 0.015, "height_m": 0.12},
    {"label": "cube", "x": 0.20, "y": 0.10, "width_m": 0.04, "height_m": 0.04},
]
```

---

### Debugging Tips

**Enable Verbose Logging:**
```bash
python server/fastmcp_robot_server.py --robot niryo --verbose
```

**Add Debug Prints:**
```python
# In client code
print(f"Calling tool: {tool_name}")
print(f"Arguments: {json.dumps(arguments, indent=2)}")
```

**Use Interactive Python:**
```python
# Test components directly
from robot_environment import Environment
env = Environment(...)
objects = env.get_detected_objects()
print(objects)
```

**Check Redis Data:**
```bash
redis-cli
> KEYS *
> GET robot_camera
> GET detected_objects
```

---

### Common Development Errors

**Import Errors:**
```python
# ✅ Good
from client.fastmcp_groq_client import RobotFastMCPClient

# ❌ Bad
from fastmcp_groq_client import RobotFastMCPClient
```

**Async/Await Issues:**
```python
# ✅ Good
async def main():
    await client.connect()

asyncio.run(main())

# ❌ Bad
def main():
    client.connect()  # Missing await!
```

**Path Issues:**
```python
# ✅ Good
server_path = Path(__file__).parent.parent / "server" / "fastmcp_robot_server.py"

# ❌ Bad
server_path = "server/fastmcp_robot_server.py"  # Relative to cwd
```

---

## Getting Help

### Information to Include in Bug Reports

1. **System Information:**
```bash
python --version
pip list | grep -E "fastmcp|groq|robot-environment"
uname -a  # Linux/Mac
```

2. **Error Messages:**
```bash
# Full stack trace
# Server logs: log/mcp_server_*.log
# Client output
```

3. **Reproduction Steps:**
```
1. Start server with: python server/...
2. Run client command: ...
3. Expected: ...
4. Actual: ...
```

4. **Configuration:**
```python
# secrets.env (without actual keys)
# Server startup command
# Client configuration
```

---

### Resources

- **GitHub Issues:** https://github.com/dgaida/robot_mcp/issues
- **Architecture:** [Architecture Guide](mcp_api_reference.md#system-architecture)
- **Examples:** [docs/examples.md](examples.md)
- **API Reference:** [docs/api.md](api.md)

---

## Quick Diagnostic Checklist

Before opening an issue, check:

- [ ] Redis is running
- [ ] Groq API key is valid
- [ ] Server is started and listening on port 8000
- [ ] Client can connect (test with curl)
- [ ] Robot is powered on (if using real robot)
- [ ] Camera is working (check Redis stream)
- [ ] Object detection is running (check for detections)
- [ ] Coordinates are within workspace bounds
- [ ] Object names match detected labels exactly
- [ ] All dependencies are installed
- [ ] Log files checked for errors

If all checked and still having issues, please open a GitHub issue with the information above!

# Configuration Management System

Centralized configuration for the Robot MCP system using YAML files, Pydantic validation, and environment-specific overrides.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration File](#configuration-file)
- [Usage](#usage)
- [Environment Overrides](#environment-overrides)
- [Command-Line Overrides](#command-line-overrides)
- [Accessing Configuration](#accessing-configuration)
- [Validation](#validation)
- [Best Practices](#best-practices)

---

## 🎯 Overview

The configuration system provides:

✅ **Type-safe configuration** - Pydantic models ensure all values are valid
✅ **Centralized settings** - No more scattered command-line args
✅ **Environment support** - dev/prod/test configs with overrides
✅ **Easy access** - Dot notation for nested values
✅ **Runtime updates** - Change settings programmatically
✅ **Validation** - Catch errors before runtime

### Before (Scattered Configuration)

```bash
# Command-line arguments
python server/fastmcp_robot_server.py --robot niryo --no-simulation --port 8000 --verbose

# Environment variables
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk-...
export ELEVENLABS_API_KEY=...

# Hardcoded defaults in multiple files
# robot_id = "niryo"
# use_simulation = True
# temperature = 0.7
```

### After (Centralized Configuration)

```yaml
# config/robot_config.yaml
server:
  port: 8000
robot:
  type: "niryo"
  simulation: false
llm:
  temperature: 0.7
```

```bash
# Clean command line
python server/fastmcp_robot_server_with_config.py

# Or with environment
ROBOT_ENV=production python server/fastmcp_robot_server_with_config.py
```

---

## 🚀 Quick Start

### 1. Create Configuration File

```bash
# Copy the template
cp config/robot_config.yaml.template config/robot_config.yaml

# Edit for your setup
nano config/robot_config.yaml
```

### 2. Set API Keys

```bash
# Keep sensitive keys in .env file
echo "OPENAI_API_KEY=sk-..." >> secrets.env
echo "GROQ_API_KEY=gsk-..." >> secrets.env
```

### 3. Run with Config

```bash
# Server uses config automatically
python server/fastmcp_robot_server_with_config.py

# Client uses config automatically
python client/fastmcp_universal_client_with_config.py
```

---

## 📄 Configuration File

### Structure

```yaml
server:          # Server settings
  host: "127.0.0.1"
  port: 8000

robot:           # Robot configuration
  type: "niryo"
  simulation: true
  workspace:     # Coordinate boundaries
    niryo:
      bounds:
        x_min: 0.163
        x_max: 0.337
  motion:        # Motion parameters
    pick_z_offset: 0.001

detection:       # Object detection
  model: "owlv2"
  confidence_threshold: 0.15
  default_labels:
    - "pencil"
    - "cube"

llm:             # LLM settings
  default_provider: "groq"
  temperature: 0.7
  providers:
    openai:
      models: ["gpt-4o", "gpt-4o-mini"]
    groq:
      models: ["llama-3.3-70b-versatile"]

tts:             # Text-to-speech
  enabled: true
  provider: "elevenlabs"

redis:           # Redis connection
  host: "localhost"
  port: 6379

gui:             # Web interface
  port: 7860
  enable_voice_input: true

logging:         # Logging config
  format: "%(asctime)s - %(levelname)s - %(message)s"
  levels:
    root: "INFO"

environments:    # Environment overrides
  production:
    robot:
      simulation: false
    server:
      host: "0.0.0.0"
```

### Complete Example

See `config/robot_config.yaml` (created in first artifact) for the complete configuration with all options documented.

---

## 💻 Usage

### In Server

```python
from config.config_manager import load_config

# Load configuration
config = load_config()

# Access settings
host = config.server.host
port = config.server.port
robot_type = config.robot.type

print(f"Starting server at {host}:{port}")
print(f"Robot: {robot_type}")
```

### In Client

```python
from config.config_manager import ConfigManager

# Load config
config = ConfigManager.load()

# Use in client initialization
client = RobotUniversalMCPClient(
    config=config,
    api_choice=config.llm.default_provider
)

# Access nested values
temperature = config.llm.temperature
max_tokens = config.llm.max_tokens
```

### Dot Notation Access

```python
# Get nested values with dot notation
x_min = config.get("robot.workspace.niryo.bounds.x_min")
# Returns: 0.163

# Get with default if not found
timeout = config.get("server.timeout", default=30)
# Returns: 30 (or configured value)

# Set values at runtime
config.set("llm.temperature", 0.9)
config.set("robot.verbose", True)
```

### Direct Property Access

```python
# Access via properties
server_config = config.server
robot_config = config.robot
llm_config = config.llm

# Properties are type-safe
print(f"Port: {server_config.port}")  # int
print(f"Simulation: {robot_config.simulation}")  # bool
print(f"Temperature: {llm_config.temperature}")  # float
```

---

## 🌍 Environment Overrides

Use environment-specific configs for dev/prod/test without changing the main config.

### Define Environments

```yaml
# config/robot_config.yaml
environments:
  development:
    server:
      log_level: "DEBUG"
    robot:
      simulation: true
      verbose: true
    llm:
      temperature: 0.8

  production:
    server:
      log_level: "WARNING"
      host: "0.0.0.0"  # Allow external connections
    robot:
      simulation: false
      verbose: false
    llm:
      temperature: 0.5  # More deterministic

  testing:
    server:
      log_level: "DEBUG"
    robot:
      enable_camera: false  # Faster tests
    llm:
      max_iterations: 5
```

### Activate Environment

```bash
# Method 1: Environment variable
export ROBOT_ENV=production
python server/fastmcp_robot_server_with_config.py

# Method 2: Command-line argument
python server/fastmcp_robot_server_with_config.py --environment production

# Method 3: Programmatically
config = ConfigManager.load(environment="production")
```

### How It Works

1. Base config is loaded from `robot_config.yaml`
2. Environment overrides are merged on top
3. Only specified values are overridden
4. Unspecified values keep base config values

**Example:**

```yaml
# Base config
robot:
  type: "niryo"
  simulation: true
  verbose: false

# Production override
environments:
  production:
    robot:
      simulation: false  # Override
      # verbose stays false (not overridden)
```

Result in production:
```python
config.robot.type       # "niryo" (from base)
config.robot.simulation # False (overridden)
config.robot.verbose    # False (from base)
```

---

## ⚙️ Command-Line Overrides

Override specific settings without modifying config files.

### Server Overrides

```bash
# Override server settings
python server/fastmcp_robot_server_with_config.py --host 0.0.0.0 --port 8080

# Override robot settings
python server/fastmcp_robot_server_with_config.py --robot-type widowx --no-simulation

# Multiple overrides
python server/fastmcp_robot_server_with_config.py \
  --robot-type niryo \
  --no-simulation \
  --no-camera \
  --verbose \
  --port 8080
```

### Client Overrides

```bash
# Override LLM provider
python client/fastmcp_universal_client_with_config.py --api openai

# Override model
python client/fastmcp_universal_client_with_config.py --api groq --model llama-3.1-8b-instant

# Single command with overrides
python client/fastmcp_universal_client_with_config.py \
  --api openai \
  --model gpt-4o \
  --command "What objects do you see?"
```

### Priority Order

1. **Command-line arguments** (highest priority)
2. **Environment overrides**
3. **Base configuration**
4. **Default values** (lowest priority)

**Example:**

```yaml
# config/robot_config.yaml
robot:
  type: "niryo"  # Base: niryo

environments:
  production:
    robot:
      type: "widowx"  # Production: widowx
```

```bash
# Command line overrides all
python server/fastmcp_robot_server_with_config.py --robot-type niryo

# Result: niryo (CLI wins)
```

---

## 🔍 Accessing Configuration

### In Application Code

```python
from config.config_manager import ConfigManager

# Get singleton instance (after load)
config = ConfigManager.get_instance()

# Or load directly
config = ConfigManager.load()

# Access server config
print(f"Server: {config.server.host}:{config.server.port}")
print(f"Log level: {config.server.log_level}")

# Access robot config
print(f"Robot: {config.robot.type}")
print(f"Simulation: {config.robot.simulation}")

# Access workspace bounds
workspace = config.robot.workspace["niryo"]
print(f"X range: [{workspace.bounds.x_min}, {workspace.bounds.x_max}]")
print(f"Y range: [{workspace.bounds.y_min}, {workspace.bounds.y_max}]")

# Access motion parameters
print(f"Pick z-offset: {config.robot.motion.pick_z_offset}")
print(f"Safe height: {config.robot.motion.safe_height}")

# Access LLM settings
print(f"Provider: {config.llm.default_provider}")
print(f"Temperature: {config.llm.temperature}")
print(f"Max tokens: {config.llm.max_tokens}")

# Get provider-specific settings
openai_config = config.llm.providers["openai"]
print(f"OpenAI models: {openai_config.models}")
print(f"Rate limit: {openai_config.rate_limit_rpm}")
```

### Runtime Modifications

```python
# Change settings at runtime
config.set("llm.temperature", 0.9)
config.set("robot.verbose", True)

# Verify changes
print(config.get("llm.temperature"))  # 0.9
print(config.robot.verbose)  # True

# Save modified config
config.save("config/my_modified_config.yaml")
```

### Export Configuration

```python
# Export as dictionary
config_dict = config.to_dict()

# Save to new file
config.save("config/backup_config.yaml")

# Pretty print
import json
print(json.dumps(config.to_dict(), indent=2))
```

---

## ✅ Validation

All configuration is validated using Pydantic models.

### Automatic Validation

```python
# This will raise ValidationError
config_dict = {
    "server": {
        "port": 99999  # ❌ Invalid: must be 1024-65535
    },
    "robot": {
        "type": "invalid"  # ❌ Invalid: must be niryo or widowx
    },
    "llm": {
        "temperature": 3.0  # ❌ Invalid: must be 0.0-2.0
    }
}

try:
    config = RobotMCPConfig(**config_dict)
except ValidationError as e:
    print("Validation errors:")
    for error in e.errors():
        print(f"  • {error['loc']}: {error['msg']}")
```

### Validation Rules

**Server:**
- `port`: 1024-65535
- `log_level`: DEBUG, INFO, WARNING, ERROR, CRITICAL

**Robot:**
- `type`: "niryo" or "widowx"
- `camera_update_rate_hz`: 0.1-30.0
- Motion parameters: sensible ranges (e.g., z_offset: 0.0-0.1m)

**Detection:**
- `model`: "owlv2" or "yoloworld"
- `device`: "cuda" or "cpu"
- `confidence_threshold`: 0.0-1.0
- `iou_threshold`: 0.0-1.0

**LLM:**
- `default_provider`: "auto", "openai", "groq", "gemini", "ollama"
- `temperature`: 0.0-2.0
- `max_tokens`: 1-128000
- `max_iterations`: 1-50

**Workspace Bounds:**
- `x_max` must be > `x_min`
- `y_max` must be > `y_min`
- `center` must be [x, y] (2 values)

### Custom Validation

Add custom validators in `config_manager.py`:

```python
class ServerConfig(BaseModel):
    port: int = Field(8000, ge=1024, le=65535)

    @field_validator("port")
    @classmethod
    def validate_port_available(cls, v: int) -> int:
        # Check if port is available
        import socket
        try:
            sock = socket.socket()
            sock.bind(('127.0.0.1', v))
            sock.close()
            return v
        except OSError:
            raise ValueError(f"Port {v} is already in use")
```

---

## 🎯 Best Practices

### 1. Keep Secrets Separate

**❌ Don't** put API keys in config files:

```yaml
# ❌ BAD
llm:
  api_key: "sk-123456789..."  # Don't do this!
```

**✅ Do** use environment variables:

```yaml
# ✅ GOOD
llm:
  default_provider: "openai"
```

```bash
# secrets.env
OPENAI_API_KEY=sk-123456789...
```

### 2. Use Environment Overrides

**❌ Don't** duplicate configs:

```
config/
  dev_config.yaml
  prod_config.yaml  # Duplication!
  test_config.yaml
```

**✅ Do** use environment overrides:

```yaml
# Single config file with overrides
environments:
  development: { ... }
  production: { ... }
  testing: { ... }
```

### 3. Validate Early

**❌ Don't** discover errors at runtime:

```python
# Program crashes after 10 minutes
robot_type = config["robt"]["tipe"]  # Typo!
```

**✅ Do** validate on load:

```python
# Fails immediately with clear error
config = ConfigManager.load()
# ValidationError: Field 'robot.type' is required
```

### 4. Use Type-Safe Access

**❌ Don't** use dictionary access:

```python
port = config.to_dict()["server"]["port"]  # No type checking
```

**✅ Do** use properties:

```python
port: int = config.server.port  # Type-safe!
```

### 5. Document Configuration

```yaml
# ✅ Add comments explaining options
detection:
  # Object detection model
  # Options: "owlv2" (accurate, slow) or "yoloworld" (fast, less accurate)
  model: "owlv2"

  # Minimum confidence score (0.0-1.0)
  # Lower = more detections but more false positives
  confidence_threshold: 0.15
```

### 6. Version Your Config

```bash
# Track config changes in git
git add config/robot_config.yaml
git commit -m "Update detection confidence threshold"

# But exclude sensitive files
echo "secrets.env" >> .gitignore
```

### 7. Provide Defaults

```python
# Always provide sensible defaults
class ServerConfig(BaseModel):
    host: str = Field("127.0.0.1", description="Server host")
    port: int = Field(8000, ge=1024, le=65535)
    max_workers: int = Field(4, ge=1)
```

---

## 🔧 Troubleshooting

### Config File Not Found

```
❌ Configuration file not found: config/robot_config.yaml
```

**Solution:**
```bash
# Create from template
cp config/robot_config.yaml.template config/robot_config.yaml

# Or specify path
python server/fastmcp_robot_server_with_config.py --config /path/to/config.yaml
```

### Validation Errors

```
❌ Configuration validation failed:
  • server.port: must be between 1024 and 65535
  • robot.type: must be one of: niryo, widowx
```

**Solution:** Fix the invalid values in your config file.

### Environment Not Applied

```
ROBOT_ENV=production python server/fastmcp_robot_server_with_config.py
# Still using development settings?
```

**Solution:** Check environment name spelling and that overrides are defined:

```yaml
environments:
  production:  # Must match ROBOT_ENV exactly
    server:
      log_level: "WARNING"
```

### Import Errors

```
ModuleNotFoundError: No module named 'config.config_manager'
```

**Solution:** Ensure you're running from project root:

```bash
# From project root
python server/fastmcp_robot_server_with_config.py

# Not from subdirectory
cd server && python fastmcp_robot_server_with_config.py  # ❌ Won't work
```

---

## 📚 Additional Resources

- **Configuration Reference**: See `config/robot_config.yaml` for all options
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **YAML Syntax**: https://yaml.org/
- **Server Documentation**: `docs/mcp_setup_guide.md`
- **API Reference**: `docs/mcp_api_reference.md`

---

## 🎉 Summary

The configuration management system provides:

✅ **Single source of truth** for all settings
✅ **Type-safe** with Pydantic validation
✅ **Environment-specific** configs (dev/prod/test)
✅ **Easy access** via dot notation and properties
✅ **Runtime updates** when needed
✅ **Clean CLI** without dozens of arguments
✅ **Better maintainability** - change once, apply everywhere

**Before:**
```bash
python server.py --host 0.0.0.0 --port 8000 --robot niryo --no-sim --camera-rate 2 --model owlv2 --confidence 0.15 --temp 0.7 --max-tokens 4096 ...
```

**After:**
```bash
python server.py  # All settings in config
```

**Happy configuring! 🚀**

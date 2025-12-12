# Communicative Robot MCP System

This enhanced version of the robot MCP system adds **LLM-generated explanations** before every tool execution, making the robot more transparent, user-friendly, and educational.

## 🎯 Features

### ✅ What's New

- **LLM-Generated Explanations**: The robot explains what it's about to do before each action
- **Intelligent Voice Output**: Only speaks for important operations (configurable)
- **Comprehensive Logging**: All explanations are logged for debugging and analysis
- **Configurable Priority System**: Control which operations get voice output
- **Multiple LLM Support**: Works with OpenAI, Groq, Gemini, or Ollama
- **Fallback System**: Graceful degradation when LLM is unavailable
- **Personality Customization**: Adjust tone, verbosity, and emoji usage

### 🤖 Example Interaction

**Without Explanations:**
```
[Robot silently picks up object]
[Robot silently places object]
```

**With Explanations:**
```
🤖 "I'm moving to observe the workspace so I can see all the objects clearly."
[Robot moves to observation pose]

🔍 "Let me scan the workspace and identify what's available."
[Robot detects objects]

🤖 "I'll pick up the pencil and place it right next to the red cube."
[Robot executes pick-and-place]

✓ "Done! The pencil is now positioned next to the red cube."
```

---

## 🚀 Quick Start

### 1. Installation

The communicative system requires `llm_client` for generating explanations:

```bash
# Ensure llm_client is installed
cd ../llm_client
pip install -e .

# Or install from GitHub
pip install git+https://github.com/dgaida/llm_client.git
```

### 2. Configure API Keys

Add your preferred LLM provider key to `secrets.env`:

```bash
# For fast, free explanations (recommended)
GROQ_API_KEY=gsk-xxxxxxxx

# Or use OpenAI
OPENAI_API_KEY=sk-xxxxxxxx

# Or use Gemini
GEMINI_API_KEY=AIzaSy-xxxxxxxx

# Ollama requires no API key (runs locally)
```

### 3. Start the Communicative Server

```bash
# Start with default settings (Groq)
python server/fastmcp_robot_server_communicative.py --robot niryo

# Use OpenAI for explanations
python server/fastmcp_robot_server_communicative.py --robot niryo --explanation-api openai

# Disable explanations (revert to silent mode)
python server/fastmcp_robot_server_communicative.py --robot niryo --no-explanations

# Use simulation mode
python server/fastmcp_robot_server_communicative.py --robot niryo --no-simulation
```

### 4. Use with MCP Client

```bash
# In another terminal
python client/fastmcp_universal_client.py
```

The client will now receive spoken explanations for important operations!

---

## ⚙️ Configuration

### Priority Levels

Control when the robot speaks using `config/explanation_config.yaml`:

```yaml
voice:
  priority:
    pick_place_object: high    # Always speak
    move2observation_pose: medium  # Speak 50% of time
    get_detected_object: low    # Speak 10% of time
    speak: never                # Never announce
```

**Priority Levels:**
- `high`: Always use voice (100%)
- `medium`: Sometimes use voice (50%)
- `low`: Rarely use voice (10%)
- `never`: Never use voice (0%)

### Personality Settings

Customize the robot's communication style:

```yaml
personality:
  use_emojis: true
  tone: "friendly"  # friendly, professional, playful
  verbosity: "concise"  # concise, detailed
```

**Tone Examples:**

**Friendly:**
> "I'm moving to observe the workspace so I can see all the objects clearly."

**Professional:**
> "Initiating observation pose transition to enable comprehensive workspace analysis."

**Playful:**
> "Time to get a bird's-eye view of the workspace! Moving into position now!"

### LLM Provider Selection

Choose your preferred LLM for generating explanations:

```yaml
llm:
  provider: "groq"  # Fast and free
  # provider: "openai"  # High quality
  # provider: "gemini"  # Good balance
  # provider: "ollama"  # Fully local

  temperature: 0.7
  max_tokens: 150
```

---

## 📊 Understanding the System

### How It Works

```
1. MCP Client sends tool call
   ↓
2. Server receives request
   ↓
3. ExplanationGenerator creates natural language explanation
   ↓
4. Priority system decides: Should we speak?
   ↓
5. If YES: Send to text2speech
   ↓
6. Log explanation
   ↓
7. Execute tool
   ↓
8. Return result with explanation context
```

### Priority Decision Logic

```python
def should_speak(tool_name):
    priority = VOICE_PRIORITY[tool_name]

    if priority == 'high':
        return True  # Always speak
    elif priority == 'medium':
        return random() > 0.5  # 50% chance
    elif priority == 'low':
        return random() > 0.9  # 10% chance
    else:
        return False  # Never speak
```

---

## 🎨 Customization Examples

### Example 1: Make Robot More Talkative

Increase voice output frequency in `explanation_config.yaml`:

```yaml
voice:
  priority:
    # Make all operations at least medium priority
    get_detected_objects: high      # Was: medium
    get_largest_detected_object: medium  # Was: low
    get_workspace_coordinate_from_point: medium  # Was: low
```

### Example 2: Silent Mode with Logs

Keep explanations in logs but disable voice:

```yaml
voice:
  enabled: false  # No speech

logging:
  log_explanations: true  # Still log everything
```

### Example 3: Detailed Explanations

Get more verbose explanations:

```yaml
personality:
  verbosity: "detailed"  # 2-4 sentences instead of 1-2

llm:
  max_tokens: 300  # Allow longer responses
```

### Example 4: Multilingual Support

Generate explanations in different languages:

```yaml
advanced:
  language: "de"  # German
  multilingual_explanations: true

prompts:
  system_base: |
    Du bist ein hilfreicher Roboter-Assistent, der seine Aktionen
    auf Deutsch erklärt. Halte Erklärungen kurz und freundlich.
```

---

## 🔍 Monitoring and Debugging

### Log Files

All explanations are logged to `log/mcp_server_YYYYMMDD_HHMMSS.log`:

```
2025-12-12 10:30:45 - TOOL CALL: pick_place_object
  Kwargs: {'object_name': 'pencil', 'pick_coordinate': [0.15, -0.05], ...}
  Explanation: 🤖 I'm picking up the pencil and placing it right next to the red cube.
  Status: SUCCESS
  Result: ✓ Successfully picked 'pencil' from [0.150, -0.050] and placed it...
```

### Real-Time Monitoring

```bash
# Watch logs in real-time
tail -f log/mcp_server_*.log | grep "Explanation:"

# Filter for specific tool
tail -f log/mcp_server_*.log | grep "pick_place_object"

# Watch voice output
tail -f log/mcp_server_*.log | grep "should_speak"
```

### Performance Metrics

Check explanation generation performance:

```bash
# Count explanations generated
grep "Explanation:" log/mcp_server_*.log | wc -l

# Find slow LLM calls
grep "Failed to generate explanation" log/mcp_server_*.log
```

---

## 🧪 Testing

### Test Explanation Generation

```python
from explanation_generator import ExplanationGenerator

# Initialize
gen = ExplanationGenerator(api_choice="groq")

# Test explanation
explanation = gen.generate_explanation(
    tool_name="pick_place_object",
    tool_description="Pick and place an object",
    arguments={
        "object_name": "pencil",
        "pick_coordinate": [0.15, -0.05],
        "place_coordinate": [0.20, 0.10]
    }
)

print(explanation)
# Output: "🤖 I'm picking up the pencil and placing it at the target location."
```

### Test Voice Priority

```python
gen = ExplanationGenerator()

# Test different priorities
print(gen.should_speak("pick_place_object"))  # True (high)
print(gen.should_speak("move2observation_pose"))  # ~50% (medium)
print(gen.should_speak("get_detected_object"))  # ~10% (low)
print(gen.should_speak("speak"))  # False (never)
```

---

## 🐛 Troubleshooting

### No Explanations Generated

**Problem:** Robot executes silently without explanations.

**Solutions:**
1. Check if `llm_client` is installed:
   ```bash
   python -c "from llm_client import LLMClient"
   ```

2. Verify API key is set:
   ```bash
   echo $GROQ_API_KEY
   ```

3. Check server logs for errors:
   ```bash
   tail -f log/mcp_server_*.log | grep "explanation"
   ```

4. Try with fallback mode:
   ```bash
   # Server will use simple templates if LLM fails
   python server/fastmcp_robot_server_communicative.py --verbose
   ```

### No Voice Output

**Problem:** Explanations are logged but not spoken.

**Solutions:**
1. Check if tool priority allows voice:
   ```yaml
   # In explanation_config.yaml
   voice:
     priority:
       your_tool: high  # Ensure it's at least high or medium
   ```

2. Verify text2speech is working:
   ```bash
   # Test TTS independently
   python -c "from text2speech import Text2Speech; tts = Text2Speech('', verbose=True); tts.call_text2speech_async('Test').join()"
   ```

3. Check voice is enabled:
   ```yaml
   voice:
     enabled: true
   ```

### Slow Response Times

**Problem:** Long delay before tool execution.

**Solutions:**
1. Use faster LLM provider:
   ```bash
   # Groq is fastest
   python server/fastmcp_robot_server_communicative.py --explanation-api groq
   ```

2. Reduce token limit:
   ```yaml
   llm:
     max_tokens: 100  # Shorter = faster
   ```

3. Disable for low-priority tools:
   ```yaml
   voice:
     priority:
       get_detected_object: never  # Skip explanation
   ```

---

## 📈 Performance Considerations

### LLM API Costs

**Groq (Recommended for development):**
- ✅ Free tier available
- ✅ Very fast (~0.5-1s per explanation)
- ✅ 30 requests/minute (free tier)

**OpenAI:**
- 💲 ~$0.01 per 1000 explanations (gpt-4o-mini)
- ⚡ Fast (~1-2s per explanation)
- ✅ High quality

**Ollama:**
- ✅ Free (local)
- ⚠️ Slower (2-5s per explanation)
- ✅ No internet required

### Response Time Impact

| Without Explanations | With Explanations (Groq) | With Explanations (Ollama) |
|---------------------|-------------------------|---------------------------|
| ~0.1s per tool call | ~0.6s per tool call | ~2.5s per tool call |

### Optimization Tips

1. **Cache common explanations** (future feature)
2. **Use faster models**: `gpt-4o-mini`, `llama-3.1-8b-instant`
3. **Reduce max_tokens**: 100-150 is usually sufficient
4. **Lower priority** for routine operations
5. **Disable** for debugging/testing

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Explanation Caching**: Cache identical tool calls
- [ ] **Context Awareness**: Use previous actions as context
- [ ] **User Preferences**: Per-user verbosity settings
- [ ] **Emotion/Tone Adaptation**: Adjust based on task success/failure
- [ ] **Multi-Step Explanations**: "I'll do X, then Y, then Z"
- [ ] **Question Answering**: "Why did you do that?"
- [ ] **Explanation Summarization**: Daily activity summaries
- [ ] **Voice Customization**: Different voices per robot/user

---

## 📚 References

- **LLM Client**: [github.com/dgaida/llm_client](https://github.com/dgaida/llm_client)
- **Text2Speech**: [github.com/dgaida/text2speech](https://github.com/dgaida/text2speech)
- **FastMCP**: [github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp)
- **Robot Environment**: [github.com/dgaida/robot_environment](https://github.com/dgaida/robot_environment)

---

## 🤝 Contributing

To add explanations for new tools:

1. Add priority level to `explanation_config.yaml`
2. Decorate function with `@log_tool_call_with_explanation`
3. Add fallback template (optional)
4. Test with different LLM providers

Example:
```python
@mcp.tool
@log_tool_call_with_explanation
@validate_input(YourInputModel)
def your_new_tool(param1: str) -> str:
    """Your tool description for explanation context."""
    # Your implementation
    pass
```

---

## 📄 License

MIT License - Same as robot_mcp

---

**Made with ❤️ for transparent robotic automation**

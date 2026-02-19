# System Architecture

The Robot MCP system is built on a modular architecture that separates the LLM reasoning, the MCP communication layer, and the physical robot control.

## System Overview

```mermaid
graph TD
    User([User]) <-->|Natural Language| Client[Universal MCP Client]
    Client <-->|LLM API| LLM[LLM Provider<br/>OpenAI/Groq/Gemini/Ollama]
    Client <-->|SSE / HTTP| Server[FastMCP Robot Server]
    Server <-->|Python API| Env[Robot Environment]
    Env <-->|Hardware Drivers| Robot[Physical Robot / Simulation]
    Env <-->|Redis| Vision[Vision System]
    Vision <-->|Camera Feed| Camera[Robot Camera]
```

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as MCP Client
    participant L as LLM
    participant S as MCP Server
    participant R as Robot

    U->>C: "Pick up the pencil"
    C->>L: Task + Available Tools
    L->>C: Reasoning (Chain-of-Thought)
    C->>U: Reasoning Display
    L->>C: Call get_detected_objects()
    C->>S: Call get_detected_objects()
    S->>C: JSON Objects List
    C->>L: Tool Result
    L->>C: Call pick_place_object(coords)
    C->>S: Call pick_place_object(coords)
    S->>R: Physical Move
    R-->>S: Success
    S-->>C: Success
    C->>L: Tool Result
    L->>C: Final Response
    C->>U: "I've picked up the pencil"
```

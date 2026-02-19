# Systemarchitektur

Das Robot MCP-System basiert auf einer modularen Architektur, die die LLM-Logik, die MCP-Kommunikationsschicht und die physische Robotersteuerung voneinander trennt.

## Systemübersicht

```mermaid
graph TD
    User([Benutzer]) <-->|Natürliche Sprache| Client[Universal MCP Client]
    Client <-->|LLM API| LLM[LLM Anbieter<br/>OpenAI/Groq/Gemini/Ollama]
    Client <-->|SSE / HTTP| Server[FastMCP Robot Server]
    Server <-->|Python API| Env[Robot Environment]
    Env <-->|Hardware Treiber| Robot[Physischer Roboter / Simulation]
    Env <-->|Redis| Vision[Visionssystem]
    Vision <-->|Kamera-Feed| Camera[Roboterkamera]
```

## Datenfluss

```mermaid
sequenceDiagram
    participant U as Benutzer
    participant C as MCP Client
    participant L as LLM
    participant S as MCP Server
    participant R as Roboter

    U->>C: "Hebe den Bleistift auf"
    C->>L: Aufgabe + Verfügbare Werkzeuge
    L->>C: Überlegungen (Chain-of-Thought)
    C->>U: Anzeige der Überlegungen
    L->>C: Aufruf get_detected_objects()
    C->>S: Aufruf get_detected_objects()
    S->>C: JSON Objektliste
    C->>L: Werkzeug-Ergebnis
    L->>C: Aufruf pick_place_object(coords)
    C->>S: Aufruf pick_place_object(coords)
    S->>R: Physische Bewegung
    R-->>S: Erfolg
    S-->>C: Erfolg
    C->>L: Werkzeug-Ergebnis
    L->>C: Finale Antwort
    C->>U: "Ich habe den Bleistift aufgehoben"
```

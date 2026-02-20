# MCP Robotersteuerung - Setup- & Nutzungsleitfaden

Vollständige Anleitung für die Einrichtung und Nutzung der natürlichen Sprachsteuerung für Roboter mit FastMCP und Multi-LLM-Unterstützung.

## Inhaltsverzeichnis

- [Übersicht](#übersicht)
- [Schnellstart](#schnellstart)
- [Nutzungsmodi](#nutzungsmodi)
- [Verfügbare LLM-Anbieter](#verfügbare-llm-anbieter)
- [Häufige Aufgaben](#häufige-aufgaben)
- [Fehlerbehebung](#fehlerbehebung)

---

## Übersicht

Das Robot MCP-System ermöglicht die Steuerung von Roboterarmen (Niryo Ned2, WidowX) durch natürliche Sprache mittels:

- **FastMCP-Server** - Stellt Robotersteuerungswerkzeuge über HTTP/SSE bereit.
- **Universal-Client** - Unterstützt OpenAI, Groq, Gemini und Ollama.
- **Visionssystem** - Objekterkennung in Echtzeit.
- **Web-Interface** - Gradio GUI mit Spracheingabe.

---

## Schnellstart

### 3-Schritte-Einrichtung

**Schritt 1: Abhängigkeiten installieren**

```bash
git clone https://github.com/dgaida/robot_mcp.git
cd robot_mcp
python -m venv venv
source venv/bin/activate
pip install -e .
```

**Schritt 2: API-Schlüssel konfigurieren**

```bash
cp secrets.env.template secrets.env
# secrets.env bearbeiten und mindestens einen API-Schlüssel hinzufügen:
```

**Schritt 3: System starten**

```bash
# Terminal 1: Redis starten
docker run -p 6379:6379 redis:alpine

# Terminal 2: FastMCP-Server starten
python server/fastmcp_robot_server.py --robot niryo

# Terminal 3: Universal-Client ausführen
python client/fastmcp_universal_client.py
```

---

## 📚 Nutzungsmodi

### 1. Interaktiver Chat-Modus (Standard)

```bash
python client/fastmcp_universal_client.py

Du: Welche Objekte siehst du?
🤖 Assistent: Ich sehe 3 Objekte: einen Bleistift, einen roten Würfel und ein blaues Quadrat.
```

### 2. Einmal-Befehl-Modus

```bash
python client/fastmcp_universal_client.py --command "Sortiere die Objekte nach Größe"
```

### 3. Gradio Web-Interface

```bash
python robot_gui/mcp_app.py --robot niryo
# Öffnen Sie http://localhost:7860 im Browser
```

---

## Verfügbare LLM-Anbieter

| Anbieter | Geschwindigkeit | Am besten für |
|----------|-----------------|---------------|
| **OpenAI** | Schnell | Komplexe Aufgaben |
| **Groq** | Sehr schnell | Entwicklung & Prototyping |
| **Gemini** | Schnell | Multimodale Aufgaben |
| **Ollama** | Variabel | Lokale Nutzung & Datenschutz |

---

## Häufige Aufgaben

- **Arbeitsbereich scannen**: "Welche Objekte siehst du?"
- **Einfaches Greifen & Platzieren**: "Hebe den Bleistift auf und lege ihn bei [0.2, 0.1] ab."
- **Relative Platzierung**: "Bewege den roten Würfel rechts neben das blaue Quadrat."

---

## Fehlerbehebung

### Server startet nicht
- Prüfen Sie, ob Port 8000 frei ist (`lsof -i :8000`).
- Stellen Sie sicher, dass Redis läuft (`redis-cli ping`).

### Client kann keine Verbindung herstellen
- Überprüfen Sie, ob der Server läuft (`curl http://127.0.0.1:8000/sse`).
- Kontrollieren Sie die API-Schlüssel in `secrets.env`.

---

**Bereit? → Starten Sie mit der [Installation](installation.md)! 🚀**

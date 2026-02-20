# Installationsanleitung - Robot MCP v0.3.0

Vollständige Installationsanleitung für das Robot MCP-Steuerungssystem mit Multi-LLM-Unterstützung.

## 📋 Voraussetzungen

### Systemanforderungen

- **Python**: 3.8 oder höher
- **Betriebssystem**: Linux, macOS oder Windows
- **RAM**: Minimum 4 GB (8 GB empfohlen)
- **GPU**: Optional, aber empfohlen für bessere Leistung
- **Redis**: Erforderlich für Kamera-Streaming und Kommunikation

### Software-Voraussetzungen

```bash
# Python-Version prüfen
python --version  # oder python3 --version

# pip aktualisieren
pip install --upgrade pip

# Git (für die Entwicklung)
git --version
```

## 🚀 Installation

### Option 1: Standard-Installation (empfohlen)

```bash
# 1. Repository klonen (falls noch nicht geschehen)
git clone https://github.com/dgaida/robot_mcp.git
cd robot_mcp

# 2. Virtuelle Umgebung erstellen
python -m venv venv

# 3. Virtuelle Umgebung aktivieren
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Paket mit allen Abhängigkeiten installieren
pip install -e ".[all]"
```

### Option 2: Minimale Installation

Nur MCP-Server und -Client, ohne GUI:

```bash
pip install -e "."
```

### Option 3: Benutzerdefinierte Installation

Wählen Sie nur die Komponenten aus, die Sie benötigen:

```bash
# Nur GUI-Komponenten
pip install -e ".[gui]"

# Entwicklungswerkzeuge
pip install -e ".[dev]"

# Dokumentationswerkzeuge
pip install -e ".[docs]"

# Alles
pip install -e ".[all]"
```

## 🔑 API-Schlüssel-Konfiguration

Das System unterstützt nun **4 LLM-Anbieter**. Sie benötigen **mindestens einen API-Schlüssel** (oder verwenden Sie Ollama für den lokalen/Offline-Betrieb).

### 1. API-Schlüssel erhalten

**OpenAI** (GPT-4o, GPT-4o-mini): [platform.openai.com](https://platform.openai.com/api-keys)
**Groq** (Llama, Mixtral) - **Kostenlose Stufe verfügbar**: [console.groq.com](https://console.groq.com/keys)
**Google Gemini** (Gemini 2.0, 2.5): [aistudio.google.com](https://aistudio.google.com/apikey)
**Ollama** (Lokale Modelle): [ollama.ai](https://ollama.ai/)

### 2. Schlüssel in secrets.env speichern

```bash
# Vorlage kopieren
cp secrets.env.template secrets.env

# Datei bearbeiten
nano secrets.env
```

**Inhalt von secrets.env:**

```bash
# OpenAI
OPENAI_API_KEY=sk-xxxx...

# Groq - EMPFOHLEN FÜR DEN EINSTIEG
GROQ_API_KEY=gsk-xxxx...

# Google Gemini
GEMINI_API_KEY=AIzaSy...
```

## 🔧 Überprüfung der Installation

### 1. Python-Imports testen

```python
import fastmcp
from llm_client import LLMClient
from robot_environment import Environment
import gradio
print("✓ Alle Core-Imports erfolgreich!")
```

### 2. Redis-Verbindung prüfen

```bash
redis-cli ping
# Erwartete Ausgabe: PONG
```

## 🛠️ Häufige Installationsprobleme

| Problem | Lösung |
|---------|--------|
| ModuleNotFoundError | `pip install -e ".[all]" --force-reinstall` |
| Redis-Verbindungsfehler | `docker run -p 6379:6379 redis:alpine` |
| API-Schlüssel nicht gefunden | Prüfen Sie, ob `secrets.env` im Projektstammverzeichnis liegt. |

---

**Installation abgeschlossen? → Weiter mit [Erste Schritte](getting-started.md)! 🚀**

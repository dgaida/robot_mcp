import os
import re
import subprocess
from datetime import datetime


def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def get_interrogate_coverage():
    stdout, stderr, code = run_command("interrogate .")
    # Search for TOTAL ... Cover%
    match = re.search(r"TOTAL\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)%", stdout)
    if match:
        return match.group(1)

    # Alternative parsing for the bottom line
    match = re.search(r"actual: ([\d.]+)%", stdout)
    if match:
        return match.group(1)

    return "0.0"

def get_ruff_status():
    stdout, stderr, code = run_command("ruff check .")
    if code == 0:
        return "✅ Passing"
    # Count lines that look like errors
    error_count = len([line for line in stdout.splitlines() if ":" in line and ".py" in line])
    if error_count == 0 and code != 0:
         return "❌ Failed"
    return f"❌ {error_count} issues"

def get_pytest_coverage():
    # Run pytest with a timeout or just skip if it takes too long
    stdout, stderr, code = run_command("pytest --cov=client --cov=server --cov=robot_gui tests/ --durations=5")
    match = re.search(r"TOTAL\s+\d+\s+\d+\s+([\d.]+)%", stdout)
    if match:
        return match.group(1)
    return "0.0"

def generate_dashboard():
    api_cov = get_interrogate_coverage()
    test_cov = get_pytest_coverage()
    lint_status = get_ruff_status()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_status_emoji(value, threshold):
        try:
            if float(value) >= threshold:
                return "✅"
            return "⚠️"
        except Exception:
            return "❓"

    markdown = f"""# Documentation & Quality Metrics

Last updated: {timestamp}

| Metric | Value | Status |
| :--- | :--- | :--- |
| API Doc Coverage | {api_cov}% | {get_status_emoji(api_cov, 95)} |
| Code Linting | {lint_status} | |
| Test Coverage | {test_cov}% | {get_status_emoji(test_cov, 80)} |

## Detailed Reports

- **API Documentation**: Uses `interrogate` to ensure all public APIs are documented.
- **Linting**: Uses `ruff` for fast Python linting and code style enforcement.
- **Testing**: Uses `pytest` and `pytest-cov` for automated testing and coverage reports.

## CI Integration

These metrics are automatically updated on every push to the main branch.
"""

    os.makedirs("docs/en/development", exist_ok=True)
    os.makedirs("docs/de/development", exist_ok=True)

    with open("docs/en/development/metrics.md", "w") as f:
        f.write(markdown)

    german_markdown = markdown.replace("# Documentation & Quality Metrics", "# Dokumentations- & Qualitätsmetriken")
    german_markdown = german_markdown.replace("Last updated", "Zuletzt aktualisiert")
    german_markdown = german_markdown.replace("| Metric | Value | Status |", "| Metrik | Wert | Status |")
    german_markdown = german_markdown.replace("API Doc Coverage", "API-Doku-Abdeckung")
    german_markdown = german_markdown.replace("Code Linting", "Code-Linting")
    german_markdown = german_markdown.replace("Test Coverage", "Testabdeckung")
    german_markdown = german_markdown.replace("Detailed Reports", "Detaillierte Berichte")
    german_markdown = german_markdown.replace("CI Integration", "CI-Integration")
    german_markdown = german_markdown.replace("These metrics are automatically updated on every push to the main branch.", "Diese Metriken werden bei jedem Push in den Main-Branch automatisch aktualisiert.")

    with open("docs/de/development/metrics.md", "w") as f:
        f.write(german_markdown)

if __name__ == "__main__":
    generate_dashboard()

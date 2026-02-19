# Veröffentlichung & Versionierung

Wir verwenden `mike`, um versionierte Dokumentation zu verwalten, und `git-cliff` für die Generierung des Versionsverlaufs.

## Dokumentations-Versionierung

So stellen Sie eine neue Version der Dokumentation bereit:

```bash
# Neue Version bereitstellen
mike deploy v0.3.0 latest --update-aliases

# Standardversion festlegen
mike set-default latest
```

## Generierung des Versionsverlaufs

Die Datei `CHANGELOG.md` wird automatisch mit `git-cliff` basierend auf Conventional Commits aktualisiert.

Manuelle Generierung:
```bash
git-cliff -o CHANGELOG.md
```

## Release-Workflow

1. Version in `pyproject.toml` aktualisieren.
2. Versionsverlauf generieren.
3. Commit und Tag erstellen.
4. Tags zu GitHub pushen.
5. Die CI wird automatisch die Dokumentation für das neue Tag erstellen und bereitstellen.

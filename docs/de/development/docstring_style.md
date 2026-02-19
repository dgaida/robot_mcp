# Docstring-Styleguide

Dieses Projekt folgt dem **Google Python Style Guide** für Docstrings.

## Format

Alle öffentlichen Module, Klassen und Methoden sollten Docstrings haben.

```python
def function_with_types_in_docstring(param1, param2):
    """Beispielfunktion mit im Docstring dokumentierten Typen.

    Args:
        param1 (int): Der erste Parameter.
        param2 (str): Der zweite Parameter.

    Returns:
        bool: Der Rückgabewert. True bei Erfolg, False andernfalls.
    """
    return True
```

## Abschnitte

### Args
Listen Sie jeden Parameter nach Namen auf. Es sollte eine Beschreibung folgen, der der Typ in Klammern vorangestellt ist.

### Returns
Beschreiben Sie den Typ und die Bedeutung des Rückgabewerts.

### Raises
Listen Sie alle relevanten Ausnahmen auf, die von der Funktion ausgelöst werden können.

### Examples
Stellen Sie Nutzungsbeispiele im Doctest-Format bereit.

## Werkzeuge
Wir verwenden `mkdocstrings` mit dem `google`-Handler, um automatisch Dokumentation aus diesen Docstrings zu generieren.

# Docstring Style Guide

This project follows the **Google Python Style Guide** for docstrings.

## Format

All public modules, classes, and methods should have docstrings.

```python
def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    return True
```

## Sections

### Args
List each parameter by name. A description should follow, preceded by the type in parentheses.

### Returns
Describe the type and meaning of the return value.

### Raises
List all relevant exceptions that can be raised by the function.

### Examples
Provide usage examples in doctest format.

## Tools
We use `mkdocstrings` with the `google` handler to automatically generate documentation from these docstrings.

# Robot MCP Tests

Comprehensive test suite for the Robot MCP control system.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_fastmcp_client.py      # FastMCP Groq client tests
├── test_fastmcp_server.py      # FastMCP robot server tests
├── test_gui.py                 # Gradio GUI tests
├── test_integration.py         # Integration tests
└── README.md                   # This file
```

## Test Categories

### Unit Tests
Test individual components in isolation with mocked dependencies.

**Files:**
- `test_fastmcp_client.py` - Client logic, tool conversion, chat processing
- `test_fastmcp_server.py` - Server tools, environment initialization
- `test_gui.py` - GUI components, event handlers, state management

### Integration Tests
Test interactions between components and end-to-end workflows.

**File:** `test_integration.py`
- Complete pick-and-place workflows
- Server-client communication
- Error recovery
- Concurrent operations
- Data flow integrity

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Files
```bash
# Client tests only
pytest tests/test_fastmcp_client.py

# Server tests only
pytest tests/test_fastmcp_server.py

# GUI tests only
pytest tests/test_gui.py

# Integration tests only
pytest tests/test_integration.py
```

### Run by Marker
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Exclude integration tests (faster for development)
pytest -m "not integration"
```

### Run with Coverage
```bash
# Generate coverage report
pytest --cov=client --cov=server --cov=robot_gui --cov-report=html

# View report in browser
open htmlcov/index.html
```

### Run with Verbose Output
```bash
pytest -v
```

### Run Specific Test
```bash
pytest tests/test_fastmcp_client.py::TestRobotFastMCPClient::test_initialization
```

## Test Fixtures

### Shared Fixtures (conftest.py)

- `test_data_dir` - Path to test data directory
- `sample_coordinates` - Sample workspace coordinates
- `sample_objects` - Sample detected objects
- `mock_groq_api_key` - Mock API key for Groq
- `mock_elevenlabs_api_key` - Mock API key for ElevenLabs

### Usage Example
```python
def test_something(sample_coordinates, sample_objects):
    center = sample_coordinates["center"]
    pencil = sample_objects[0]
    # Use fixtures in test
```

## Test Coverage

### Current Coverage Areas

✅ **Client Components**
- Client initialization
- Connection/disconnection
- Tool calling
- Chat processing
- Tool call handling
- Error handling
- System prompt

✅ **Server Components**
- Environment initialization
- Robot control tools (pick, place, push)
- Object detection tools
- Workspace tools
- Feedback tools
- Location enum handling

✅ **GUI Components**
- Initialization
- Environment setup
- MCP server/client connection
- Message handling
- Voice input
- Camera feed
- Status display
- Error handling

✅ **Integration**
- End-to-end workflows
- Server-client communication
- Tool synchronization
- Error recovery
- Concurrent operations
- Data serialization

### Coverage Goals

Target: **>80% code coverage**

Current status:
```bash
pytest --cov=client --cov=server --cov=robot_gui --cov-report=term
```

## Writing New Tests

### Unit Test Template
```python
# tests/test_new_component.py
import pytest
from unittest.mock import MagicMock, patch

class TestNewComponent:
    """Test suite for NewComponent."""

    @pytest.fixture
    def component(self):
        """Create component instance."""
        with patch('module.Dependency'):
            return NewComponent()

    def test_initialization(self, component):
        """Test component initialization."""
        assert component is not None
        assert component.state == "initialized"

    def test_operation(self, component):
        """Test component operation."""
        result = component.do_something()
        assert result == expected_value
```

### Integration Test Template
```python
# tests/test_integration.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow():
    """Test complete workflow."""
    # Setup
    client = create_client()
    await client.connect()

    # Execute
    result = await client.perform_action()

    # Verify
    assert result.success is True

    # Cleanup
    await client.disconnect()
```

## Mocking Guidelines

### External Dependencies to Mock

**Always mock:**
- `Groq` API client
- `fastmcp.Client` MCP client
- `robot_environment.Environment` robot environment
- `Speech2Text` speech recognition
- `RedisImageStreamer` image streaming
- `subprocess.Popen` for server processes

### Example Mocking
```python
with patch('module.ExternalDependency') as mock_dep:
    mock_dep.return_value = MagicMock()
    mock_dep.return_value.method.return_value = "expected_value"

    # Run test with mocked dependency
    result = function_under_test()

    assert result == "expected_value"
    mock_dep.assert_called_once()
```

## Async Testing

For async functions, use `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## Debugging Tests

### Run with Debug Output
```bash
pytest -vv -s tests/test_file.py
```

### Debug Failed Test
```bash
pytest --lf  # Run last failed
pytest --ff  # Run failed first
```

### Use pdb Debugger
```python
def test_something():
    import pdb; pdb.set_trace()
    # Test code
```

## Continuous Integration

Tests run automatically on:
- Push to master branch
- Pull requests
- GitHub Actions workflow

See `.github/workflows/tests.yml` for CI configuration.

## Common Issues

### Import Errors
**Problem:** Module not found errors
**Solution:** Ensure project root is in PYTHONPATH
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Async Warnings
**Problem:** Warnings about unclosed coroutines
**Solution:** Ensure all async fixtures are properly awaited

### Mock Not Working
**Problem:** Mock not intercepting calls
**Solution:** Check patch target path is correct
```python
# Patch where it's used, not where it's defined
patch('client.module.Dependency')  # Correct
patch('dependency.module.Dependency')  # Wrong
```

## Best Practices

1. **One assertion per test** (when practical)
2. **Descriptive test names** - `test_connection_failure_with_invalid_key`
3. **Arrange-Act-Assert pattern**
   ```python
   # Arrange
   client = setup_client()
   # Act
   result = client.connect()
   # Assert
   assert result is True
   ```
4. **Use fixtures** for common setup
5. **Mock external dependencies** - never hit real APIs in tests
6. **Test edge cases** - not just happy path
7. **Keep tests independent** - no test should depend on another

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >80% code coverage
4. Add integration tests for new workflows

## Support

For test-related questions:
- Check this README
- Review existing tests for examples
- Open an issue on GitHub

---

**Last Updated:** 2025-01-11

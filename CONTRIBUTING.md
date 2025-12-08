# Contributing to LLM Judge Auditor

Thank you for your interest in contributing to the LLM Judge Auditor toolkit!

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd llm-judge-auditor
```

### 2. Set Up Virtual Environment

**Important:** Always use a dedicated virtual environment for development.

**Quick setup (recommended):**
```bash
# macOS/Linux
./setup_env.sh

# Windows
setup_env.bat
```

**Manual setup:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests
pytest

# Check code style
black --check src tests
ruff check src tests
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
pytest --cov=llm_judge_auditor --cov-report=html

# Run property-based tests only
pytest tests/property/
```

### Code Style

We use `black` for formatting and `ruff` for linting:

```bash
# Format code
black src tests

# Check linting
ruff check src tests

# Auto-fix linting issues
ruff check --fix src tests
```

### Type Checking

```bash
mypy src
```

## Project Structure

```
llm-judge-auditor/
├── src/llm_judge_auditor/     # Main package
│   ├── components/            # Core components
│   ├── utils/                 # Utility functions
│   ├── config.py             # Configuration
│   ├── models.py             # Data models
│   └── cli.py                # CLI interface
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── property/             # Property-based tests
│   └── integration/          # Integration tests
├── config/                    # Configuration files
├── examples/                  # Usage examples
└── docs/                      # Documentation
```

## Testing Guidelines

- Write unit tests for all new components
- Add property-based tests for correctness properties
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation as needed
6. Submit a pull request

## Questions?

Feel free to open an issue for any questions or concerns.

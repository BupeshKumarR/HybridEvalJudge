# Quick Start Guide

## 1. Setup (You Already Have .venv!)

```bash
# Activate the virtual environment
source .venv/bin/activate          # macOS/Linux
# or
.venv\Scripts\activate.bat         # Windows
```

Your prompt will change from `(base)` to `(.venv)` when activated.

**Important:** Always activate the virtual environment before working on the project!

## 2. Verify Installation

```bash
# Run tests to verify everything works
pytest

# Should see: 27 passed
```

## 3. Basic Usage

```python
from llm_judge_auditor import ToolkitConfig, EvaluationRequest

# Load a preset configuration
config = ToolkitConfig.from_preset("balanced")

# Create an evaluation request
request = EvaluationRequest(
    source_text="The Eiffel Tower is in Paris, France.",
    candidate_output="The Eiffel Tower is in London.",
    task="factual_accuracy"
)

# Evaluation toolkit will be implemented in subsequent tasks
```

## 4. Run Examples

```bash
# Simple evaluation (recommended starting point)
python examples/simple_evaluation.py

# Basic usage with data models
python examples/basic_usage.py

# Batch processing
python examples/batch_processing_example.py
```

## 5. Development Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
pytest --cov=llm_judge_auditor

# Format code
make format

# Check linting
make lint
```

## 6. Daily Workflow

**Start working:**
```bash
cd llm-judge-auditor
source .venv/bin/activate  # Don't forget this!
```

**Make changes, run tests:**
```bash
pytest
```

**Stop working:**
```bash
deactivate
```

## Need Help?

- Full setup guide: `docs/ENVIRONMENT_SETUP.md`
- Contributing guide: `CONTRIBUTING.md`
- Run `make help` for available commands

## Next Steps

The project structure is now set up! The next tasks will implement:
- Device Manager (Task 2)
- Model Manager (Task 3)
- Core evaluation components (Tasks 4-12)

Check `.kiro/specs/llm-judge-auditor/tasks.md` for the full implementation plan.

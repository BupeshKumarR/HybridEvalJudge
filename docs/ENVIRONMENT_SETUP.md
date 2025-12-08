# Environment Setup Guide

## Why Use a Virtual Environment?

Using a dedicated virtual environment for this project is **strongly recommended** because:

1. **Isolation**: Keeps project dependencies separate from your system Python and other projects
2. **Reproducibility**: Ensures everyone uses the same package versions
3. **Safety**: Prevents conflicts with globally installed packages
4. **Clean uninstall**: Easy to remove by just deleting the `venv` folder

## Quick Setup

### Option 1: Automated Setup (Recommended)

**macOS/Linux:**
```bash
./setup_env.sh
source venv/bin/activate
```

**Windows:**
```bash
setup_env.bat
venv\Scripts\activate.bat
```

### Option 2: Using Make

```bash
make setup
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate.bat  # Windows

make install
```

### Option 3: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate.bat  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Verifying Your Setup

After activation, your prompt should show `(venv)` prefix:

```bash
(venv) user@machine:~/llm-judge-auditor$
```

Verify the installation:

```bash
# Check Python is from venv
which python  # macOS/Linux
where python  # Windows

# Should show: /path/to/llm-judge-auditor/venv/bin/python

# Run tests
pytest

# Try importing the package
python -c "from llm_judge_auditor import ToolkitConfig; print('Success!')"
```

## Daily Workflow

### Starting Work

```bash
# Navigate to project
cd llm-judge-auditor

# Activate environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate.bat  # Windows
```

### During Development

```bash
# Run tests
pytest

# Run specific tests
pytest tests/unit/test_config.py

# Format code
make format

# Check linting
make lint
```

### Ending Work

```bash
# Deactivate environment
deactivate
```

## IDE Configuration

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

### PyCharm

1. Go to: Settings → Project → Python Interpreter
2. Click the gear icon → Add
3. Select "Existing environment"
4. Choose: `<project-path>/venv/bin/python`

## Troubleshooting

### "Command not found: python3"

Install Python 3.9 or higher from [python.org](https://www.python.org/downloads/)

### "Permission denied: ./setup_env.sh"

Make the script executable:
```bash
chmod +x setup_env.sh
```

### "Module not found" errors

Ensure the virtual environment is activated and packages are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Wrong Python version in venv

Delete and recreate:
```bash
rm -rf venv
python3.11 -m venv venv  # Use specific Python version
source venv/bin/activate
pip install -r requirements.txt
```

## Updating Dependencies

When `requirements.txt` changes:

```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

## Removing the Environment

Simply delete the folder:

```bash
rm -rf venv
```

Then recreate when needed with `./setup_env.sh`

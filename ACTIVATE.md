# Quick Activation Guide

## Your Environment Name: `.venv`

You already have a virtual environment set up at `.venv` in your project!

## To Activate

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate.bat
```

## How to Know It's Active

Your terminal prompt will change from:
```
(base) user@machine:~/project$
```

to:
```
(.venv) user@machine:~/project$
```

## To Deactivate

```bash
deactivate
```

## Quick Test

After activating, verify everything works:
```bash
pytest
python examples/basic_usage.py
```

All 27 tests should pass! ✅

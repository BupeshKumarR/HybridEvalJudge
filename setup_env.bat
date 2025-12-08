@echo off
REM Setup script for creating a dedicated virtual environment for LLM Judge Auditor (Windows)

echo ==========================================
echo LLM Judge Auditor - Environment Setup
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment
if exist venv (
    echo Virtual environment already exists at .\venv
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
    ) else (
        echo Using existing virtual environment
        exit /b 0
    )
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo Installing package in editable mode...
pip install -e .

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo   deactivate
echo.
echo To run tests:
echo   pytest
echo.

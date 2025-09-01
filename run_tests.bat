@echo off
echo 🔧 Saksoft MFD API Testing Script
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
echo 📦 Installing/Checking requirements...
pip install -r test_requirements.txt

echo.
echo 🚀 Starting API Tests...
echo.

REM Run the test script
python test_api_complete.py

echo.
echo ✅ Tests completed!
pause







#!/bin/bash

echo "🔧 Saksoft MFD API Testing Script"
echo "==================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

# Check if requirements are installed
echo "📦 Installing/Checking requirements..."
pip3 install -r test_requirements.txt

echo
echo "🚀 Starting API Tests..."
echo

# Run the test script
python3 test_api_complete.py

echo
echo "✅ Tests completed!"







#!/bin/bash

echo "🔥 HISYNC AI - Clean YOLO12 Classification API"
echo "=============================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9 or higher."
    exit 1
fi

echo "🔄 Installing requirements..."
pip3 install -r requirements.txt

echo ""
echo "🚀 Starting YOLO12 API Server..."
echo "📝 Server will start at: http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🎯 Interactive UI: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 main.py

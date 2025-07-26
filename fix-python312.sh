#!/bin/bash

# HISYNC AI - Python 3.12 Fix Script
# Fixes the "externally-managed-environment" error in Python 3.12+

echo "🔧 HISYNC AI - Python 3.12 Fix"
echo "==============================="
echo "Fixing externally-managed-environment error..."
echo ""

# Auto-detect Python version
PYTHON_CMD=""
for py_version in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v $py_version >/dev/null 2>&1; then
        PYTHON_CMD=$py_version
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ No Python 3 found!"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "🐍 Using Python: $PYTHON_CMD ($PYTHON_VERSION)"
echo ""

# Install packages with --break-system-packages flag
echo "📦 Installing HISYNC AI packages with Python 3.12+ compatibility..."

# Core packages
PACKAGES=(
    "fastapi==0.104.1"
    "uvicorn==0.24.0"
    "pydantic==2.5.0"
    "python-multipart==0.0.6"
    "aiofiles==23.2.1"
    "tensorflow==2.15.0"
    "pillow==10.1.0"
    "numpy==1.24.3"
    "opencv-python==4.8.1.78"
    "scikit-learn==1.3.2"
    "matplotlib==3.8.2"
    "python-magic==0.4.27"
)

for package in "${PACKAGES[@]}"; do
    echo "📦 Installing $package..."
    if $PYTHON_CMD -m pip install --user --break-system-packages "$package" --quiet; then
        echo "   ✅ $package installed"
    else
        echo "   ⚠️  $package installation had issues, but continuing..."
    fi
done

echo ""
echo "🧪 Testing imports..."

# Test critical imports
IMPORT_TESTS=(
    "fastapi:FastAPI"
    "uvicorn:Uvicorn"
    "tensorflow:TensorFlow"
    "PIL:Pillow"
    "numpy:NumPy"
    "cv2:OpenCV"
)

for test in "${IMPORT_TESTS[@]}"; do
    module=$(echo "$test" | cut -d: -f1)
    name=$(echo "$test" | cut -d: -f2)
    
    if $PYTHON_CMD -c "import $module" 2>/dev/null; then
        echo "   ✅ $name"
    else
        echo "   ❌ $name - import failed"
    fi
done

echo ""
echo "🚀 Starting HISYNC AI..."

# Set Python path
export PYTHONPATH="$HOME/.local/lib/python$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"

# Kill any existing process
pkill -f "python.*main.py" 2>/dev/null || true

# Start in background
nohup $PYTHON_CMD main.py > hisync-ai.log 2>&1 &
PID=$!
echo $PID > hisync-ai.pid

echo "✅ HISYNC AI started (PID: $PID)"
echo "📝 Logs: tail -f hisync-ai.log"

# Wait and test
sleep 5

if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
    echo ""
    echo "🎉 HISYNC AI is running successfully!"
    echo "🌐 Access: http://localhost:8000"
    echo "📚 Docs: http://localhost:8000/docs"
else
    echo "❌ Health check failed"
    echo "📝 Check logs: tail -f hisync-ai.log"
fi

echo ""
echo "🔧 Management commands:"
echo "   • Check status: ps -p $PID"
echo "   • View logs: tail -f hisync-ai.log"
echo "   • Stop: kill $PID"
echo ""
echo "🔥 Python 3.12 fix complete!" 
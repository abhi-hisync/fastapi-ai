#!/bin/bash

# HISYNC AI - OpenCV Headless Fix
# Fixes the OpenGL dependency issue on headless servers

echo "🔧 HISYNC AI - OpenCV Headless Fix"
echo "=================================="
echo "Fixing OpenGL dependency issue for headless servers..."
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

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "🗑️  Uninstalling opencv-python (with GUI dependencies)..."
$PYTHON_CMD -m pip uninstall opencv-python -y --break-system-packages 2>/dev/null || true

echo "📦 Installing opencv-python-headless (no GUI dependencies)..."
$PYTHON_CMD -m pip install --user --break-system-packages opencv-python-headless --upgrade --quiet

echo "🧪 Testing OpenCV import..."
if $PYTHON_CMD -c "import cv2; print(f'✅ OpenCV {cv2.__version__} imported successfully')" 2>/dev/null; then
    echo "✅ OpenCV headless working!"
else
    echo "❌ OpenCV import still failing"
    exit 1
fi

echo "🚀 Restarting HISYNC AI..."

# Kill existing process
if [ -f "hisync-ai.pid" ]; then
    PID=$(cat hisync-ai.pid)
    kill $PID 2>/dev/null || true
    rm -f hisync-ai.pid
fi

# Set Python path
export PYTHONPATH="$HOME/.local/lib/python$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"

# Start HISYNC AI
nohup $PYTHON_CMD main.py > hisync-ai.log 2>&1 &
PID=$!
echo $PID > hisync-ai.pid

echo "✅ HISYNC AI started (PID: $PID)"

# Wait and test
sleep 10

if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
    echo ""
    echo "🎉 HISYNC AI is running successfully!"
    echo "🌐 Access: http://localhost:8000"
    echo "📚 Docs: http://localhost:8000/docs"
    echo "🏢 Company: http://localhost:8000/company"
else
    echo "❌ Health check failed"
    echo "📝 Check logs: tail -f hisync-ai.log"
fi

echo ""
echo "🔥 OpenCV headless fix complete!" 
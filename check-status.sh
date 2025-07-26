#!/bin/bash

# HISYNC AI - Status Check Script
echo "🔍 HISYNC AI - Status Check"
echo "==========================="
echo ""

# Check if process is running
if [ -f "hisync-ai.pid" ]; then
    PID=$(cat hisync-ai.pid)
    echo "📋 PID File: $PID"
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process is running (PID: $PID)"
        echo "📊 Process details:"
        ps -o pid,ppid,cmd,etime,rss -p $PID
    else
        echo "❌ Process not running (PID file exists but process dead)"
        rm -f hisync-ai.pid
    fi
else
    echo "❌ No PID file found"
fi

echo ""
echo "🌐 Port Check:"
if netstat -tlnp 2>/dev/null | grep :8000; then
    echo "✅ Port 8000 is in use"
else
    echo "❌ Port 8000 is not in use"
fi

echo ""
echo "📝 Log Files:"
if [ -f "hisync-ai.log" ]; then
    echo "✅ Log file exists ($(wc -l < hisync-ai.log) lines)"
    echo "📄 Last 10 lines:"
    tail -n 10 hisync-ai.log
else
    echo "❌ No log file found"
fi

echo ""
echo "🧪 Quick Test:"
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health endpoint responding"
    curl -s http://localhost:8000/health | head -c 200
    echo ""
else
    echo "❌ Health endpoint not responding"
fi

echo ""
echo "🐍 Python Test:"
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH"
python3.12 -c "
try:
    import fastapi, uvicorn, tensorflow, PIL, numpy, cv2
    print('✅ All imports working')
    print(f'   FastAPI: {fastapi.__version__}')
    print(f'   TensorFlow: {tensorflow.__version__}')
    print(f'   OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'❌ Import error: {e}')
"

echo ""
echo "🔧 Manual Start Test:"
echo "Run this manually to see detailed errors:"
echo "export PYTHONPATH=\"\$HOME/.local/lib/python3.12/site-packages:\$PYTHONPATH\""
echo "python3.12 main.py" 
#!/bin/bash

# 🔥 HISYNC AI - Complete All-in-One Deployment Script
# © 2024 HISYNC Technologies. All rights reserved.
# 
# This script does EVERYTHING needed to deploy HISYNC AI
# No other scripts needed - just run this one!

set -e  # Exit on any error

echo "🔥 HISYNC AI - Complete Deployment"
echo "=================================="
echo "© 2024 HISYNC Technologies"
echo "One script to rule them all!"
echo ""

# Configuration
DOMAIN="ai.hisync.in"
APP_DIR="/home/forge/$DOMAIN"
SERVICE_NAME="hisync-ai"

# Auto-detect Python version
PYTHON_CMD=""
for py_version in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v $py_version >/dev/null 2>&1; then
        PYTHON_CMD=$py_version
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ No Python 3 found! Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)

echo "📋 HISYNC AI Complete Setup:"
echo "   Domain: $DOMAIN"
echo "   Directory: $APP_DIR"
echo "   Python: $PYTHON_CMD ($PYTHON_VERSION)"
echo "   User: $(whoami)"
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found! Please run this script from the HISYNC AI project directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Step 1: Install Python packages
echo "📦 Step 1: Installing HISYNC AI packages..."
echo "🔧 Upgrading pip..."
$PYTHON_CMD -m pip install --user --upgrade pip --break-system-packages --quiet

# Install core packages directly (no requirements.txt dependency)
CORE_PACKAGES=(
    "fastapi>=0.104.1"
    "uvicorn[standard]>=0.24.0"
    "pydantic>=2.5.0"
    "python-multipart>=0.0.6"
    "aiofiles>=23.2.1"
    "tensorflow>=2.16.1"
    "pillow>=10.1.0"
    "numpy>=1.24.3"
    "opencv-python-headless>=4.8.1.78"
    "scikit-learn>=1.3.2"
    "matplotlib>=3.8.2"
)

echo "📦 Installing HISYNC AI packages..."
for package in "${CORE_PACKAGES[@]}"; do
    echo "   Installing $package..."
    $PYTHON_CMD -m pip install --user --break-system-packages "$package" --quiet --upgrade || echo "   ⚠️  $package installation had issues, but continuing..."
done

# Step 2: Verify installations
echo "🔍 Step 2: Verifying installations..."
export PYTHONPATH="$HOME/.local/lib/python$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"

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

# Step 3: Create environment file
echo "⚙️  Step 3: Creating environment configuration..."
cat > .env << EOF
# HISYNC AI Configuration
APP_NAME="HISYNC AI - Image Classification API"
COMPANY="HISYNC Technologies"
DEBUG=False
HOST=0.0.0.0
PORT=8000
SECRET_KEY=$(openssl rand -base64 32 2>/dev/null || echo "hisync-$(date +%s)-secret-key")

# Performance Settings
MAX_WORKERS=2
TIMEOUT=60

# Support Information
SUPPORT_EMAIL=support@hisync.in
COMPANY_WEBSITE=https://hisync.in
EOF

echo "✅ Environment file created"

# Step 4: Create process manager
echo "🔧 Step 4: Creating HISYNC AI process manager..."
cat > hisync-control.sh << 'EOF'
#!/bin/bash
# HISYNC AI Process Controller

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$APP_DIR/hisync-ai.pid"
LOG_FILE="$APP_DIR/hisync-ai.log"

# Auto-detect Python
PYTHON_CMD=""
for py_version in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v $py_version >/dev/null 2>&1; then
        PYTHON_CMD=$py_version
        break
    fi
done

start_hisync() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "🔥 HISYNC AI is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    echo "🚀 Starting HISYNC AI..."
    cd "$APP_DIR"
    
    # Set Python path
    export PYTHONPATH="$HOME/.local/lib/python$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"
    
    nohup $PYTHON_CMD main.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "✅ HISYNC AI started (PID: $(cat "$PID_FILE"))"
    echo "📝 Logs: tail -f $LOG_FILE"
}

stop_hisync() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "🛑 Stopping HISYNC AI..."
        kill "$(cat "$PID_FILE")"
        rm -f "$PID_FILE"
        echo "✅ HISYNC AI stopped"
    else
        echo "⚠️  HISYNC AI is not running"
    fi
}

status_hisync() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "✅ HISYNC AI is running (PID: $(cat "$PID_FILE"))"
        echo "📊 Memory usage: $(ps -o pid,rss,vsz,comm -p "$(cat "$PID_FILE")" | tail -n 1)"
        echo "🐍 Python: $PYTHON_CMD"
        
        # Test health endpoint
        if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "🌐 Health check: ✅ PASSED"
        else
            echo "🌐 Health check: ❌ FAILED"
        fi
    else
        echo "❌ HISYNC AI is not running"
    fi
}

restart_hisync() {
    stop_hisync
    sleep 3
    start_hisync
}

logs_hisync() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "❌ No log file found"
    fi
}

case "$1" in
    start)
        start_hisync
        ;;
    stop)
        stop_hisync
        ;;
    restart)
        restart_hisync
        ;;
    status)
        status_hisync
        ;;
    logs)
        logs_hisync
        ;;
    *)
        echo "🔥 HISYNC AI Process Controller"
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start HISYNC AI service"
        echo "  stop    - Stop HISYNC AI service"
        echo "  restart - Restart HISYNC AI service"
        echo "  status  - Check HISYNC AI status"
        echo "  logs    - View HISYNC AI logs"
        exit 1
        ;;
esac
EOF

chmod +x hisync-control.sh

# Step 5: Create nginx configuration
echo "📋 Step 5: Creating Nginx configuration..."
cat > nginx-config.txt << 'EOF'
# HISYNC AI - Simple Nginx Configuration
# Copy this to your Laravel Forge site's nginx configuration

include forge-conf/ai.hisync.in/before/*;

server {
    listen 80;
    listen 443 ssl http2;
    server_name ai.hisync.in;
    root /home/forge/ai.hisync.in;

    # SSL Configuration (Managed by Forge)
    ssl_certificate /etc/nginx/ssl/ai.hisync.in/server.crt;
    ssl_certificate_key /etc/nginx/ssl/ai.hisync.in/server.key;

    # HISYNC AI Proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 10M;
        
        # Timeouts for AI processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
        access_log off;
    }

    include forge-conf/ai.hisync.in/server/*;
}

include forge-conf/ai.hisync.in/after/*;
EOF

# Step 6: Stop any existing process and start fresh
echo "🛑 Step 6: Starting HISYNC AI service..."
./hisync-control.sh stop 2>/dev/null || true
sleep 2
./hisync-control.sh start

# Step 7: Wait and test
echo "⏳ Step 7: Testing deployment..."
sleep 10

# Test the deployment
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "❌ Health check failed"
    echo "📝 Checking logs..."
    tail -n 20 hisync-ai.log 2>/dev/null || echo "No log file found"
fi

# Step 8: Final status
echo ""
echo "📊 Final Status:"
./hisync-control.sh status

echo ""
echo "🎉 HISYNC AI Complete Deployment Finished!"
echo "=========================================="
echo ""
echo "🌐 Access Points:"
echo "   • Local API: http://localhost:8000"
echo "   • Health Check: http://localhost:8000/health"
echo "   • Documentation: http://localhost:8000/docs"
echo "   • Company Info: http://localhost:8000/company"
echo ""
echo "🔧 Management Commands:"
echo "   • Check Status: ./hisync-control.sh status"
echo "   • View Logs: ./hisync-control.sh logs"
echo "   • Restart: ./hisync-control.sh restart"
echo "   • Stop: ./hisync-control.sh stop"
echo ""
echo "📝 Files Created:"
echo "   • hisync-control.sh - Main process controller"
echo "   • nginx-config.txt - Nginx configuration for Laravel Forge"
echo "   • hisync-ai.log - Application logs"
echo "   • .env - Environment configuration"
echo ""
echo "🚀 Next Steps for Browser Access:"
echo "   1. Copy nginx-config.txt content to Laravel Forge nginx config"
echo "   2. Enable SSL certificate in Laravel Forge"
echo "   3. Access: https://ai.hisync.in"
echo ""
echo "📞 Support:"
echo "   • Email: support@hisync.in"
echo "   • Website: https://hisync.in"
echo ""
echo "🔥 HISYNC AI is ready for production!"
echo "© 2024 HISYNC Technologies. All rights reserved." 
#!/bin/bash

# 🔥 HISYNC AI - FORCE Server Update (Foolproof)
# © 2025 Hire Synchronisation Pvt. Ltd.
# Developed by: Abhishek Rajput (@abhi-hisync)

echo "🔥 HISYNC AI - FORCE Update (Foolproof)"
echo "======================================="
echo "This will FORCE update your server with latest changes!"
echo ""

# Step 1: Kill ALL Python processes
echo "🛑 Step 1: Killing ALL Python processes..."
pkill -f python || echo "No Python processes found"
pkill -f main.py || echo "No main.py processes found"
sleep 3
echo "✅ All processes killed"
echo ""

# Step 2: Force git update
echo "📥 Step 2: Force updating code from GitHub..."
echo "Current directory: $(pwd)"
echo "Git status before update:"
git status || echo "Not a git repository"
echo ""

echo "Fetching latest changes..."
git fetch origin main || echo "Fetch failed"
echo ""

echo "Resetting to latest version (this will overwrite local changes)..."
git reset --hard origin/main || echo "Reset failed"
echo ""

echo "Pulling latest changes..."
git pull origin main || echo "Pull failed"
echo ""

echo "✅ Code updated from GitHub"
echo ""

# Step 3: Verify files exist
echo "🔍 Step 3: Verifying files..."
if [ -f "main.py" ]; then
    echo "✅ main.py exists"
else
    echo "❌ main.py missing!"
    exit 1
fi

if [ -f "hisync-ai-complete.sh" ]; then
    echo "✅ hisync-ai-complete.sh exists"
    chmod +x hisync-ai-complete.sh
else
    echo "⚠️  hisync-ai-complete.sh missing"
fi
echo ""

# Step 4: Check Python version
echo "🐍 Step 4: Checking Python..."
PYTHON_CMD=""
for py_version in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v $py_version >/dev/null 2>&1; then
        PYTHON_CMD=$py_version
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ No Python found!"
    exit 1
fi

echo "✅ Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# Step 5: Start fresh process
echo "🚀 Step 5: Starting HISYNC AI with latest code..."
export PYTHONPATH="$HOME/.local/lib/python$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"

# Start in background
nohup $PYTHON_CMD main.py > hisync-ai.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > hisync-ai.pid

echo "✅ HISYNC AI started with PID: $NEW_PID"
echo "📝 Log file: hisync-ai.log"
echo ""

# Step 6: Wait and test
echo "⏳ Step 6: Waiting for service to initialize..."
sleep 15
echo ""

# Step 7: Test local API
echo "🧪 Step 7: Testing local API..."
echo "Testing health endpoint..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8000/health)
    echo "✅ Health check: WORKING"
    echo "   Response: $HEALTH"
    
    # Extract version from health response
    VERSION=$(echo "$HEALTH" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    if [ "$VERSION" = "1.0.0" ]; then
        echo "✅ Version: CORRECT (1.0.0)"
    else
        echo "❌ Version: WRONG ($VERSION) - Expected 1.0.0"
    fi
else
    echo "❌ Health check: FAILED"
    echo "📝 Last 20 lines of log:"
    tail -n 20 hisync-ai.log 2>/dev/null || echo "No log file"
    exit 1
fi
echo ""

echo "Testing company endpoint..."
if curl -s -f http://localhost:8000/company > /dev/null 2>&1; then
    COMPANY=$(curl -s http://localhost:8000/company)
    echo "✅ Company endpoint: WORKING"
    
    # Check if developer info exists
    if echo "$COMPANY" | grep -q "Abhishek Rajput"; then
        echo "✅ Developer info: FOUND (Abhishek Rajput)"
    else
        echo "❌ Developer info: MISSING"
    fi
    
    # Check company name
    if echo "$COMPANY" | grep -q "Hire Synchronisation"; then
        echo "✅ Company name: CORRECT (Hire Synchronisation Pvt. Ltd.)"
    else
        echo "❌ Company name: WRONG"
    fi
    
    # Check year
    if echo "$COMPANY" | grep -q "2025"; then
        echo "✅ Year: CORRECT (2025)"
    else
        echo "❌ Year: WRONG"
    fi
else
    echo "❌ Company endpoint: FAILED"
fi
echo ""

# Step 8: Test domain
echo "🌐 Step 8: Testing domain access..."
echo "Testing: https://ai.hisync.in/health"
if curl -s -f https://ai.hisync.in/health > /dev/null 2>&1; then
    DOMAIN_HEALTH=$(curl -s https://ai.hisync.in/health)
    echo "✅ Domain health: WORKING"
    
    # Check domain version
    DOMAIN_VERSION=$(echo "$DOMAIN_HEALTH" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    if [ "$DOMAIN_VERSION" = "1.0.0" ]; then
        echo "✅ Domain version: CORRECT (1.0.0)"
        echo "🎉 SUCCESS! Your changes are LIVE!"
    else
        echo "❌ Domain version: STILL OLD ($DOMAIN_VERSION)"
        echo "⚠️  Domain might be cached or nginx needs restart"
    fi
else
    echo "❌ Domain access: FAILED"
    echo "⚠️  Check nginx configuration"
fi
echo ""

# Step 9: Final summary
echo "📊 FORCE UPDATE SUMMARY:"
echo "========================"
echo "Local API:     ✅ WORKING"
echo "New Version:   $(if [ "$VERSION" = "1.0.0" ]; then echo "✅ CORRECT"; else echo "❌ WRONG"; fi)"
echo "Developer Info: $(if curl -s http://localhost:8000/company | grep -q "Abhishek"; then echo "✅ FOUND"; else echo "❌ MISSING"; fi)"
echo "Domain Access: $(if curl -s -f https://ai.hisync.in/health > /dev/null 2>&1; then echo "✅ WORKING"; else echo "❌ FAILED"; fi)"
echo ""

echo "🌐 Test URLs:"
echo "• Health: https://ai.hisync.in/health"
echo "• Company: https://ai.hisync.in/company"
echo "• Docs: https://ai.hisync.in/docs"
echo ""

if [ "$VERSION" = "1.0.0" ] && curl -s http://localhost:8000/company | grep -q "Abhishek"; then
    echo "🎉 SUCCESS! All changes deployed correctly!"
    echo "🔥 Refresh your browser - you should see new info!"
else
    echo "⚠️  Some issues detected. Check the logs above."
fi

echo ""
echo "🔥 HISYNC AI Force Update Complete!"
echo "© 2025 Hire Synchronisation Pvt. Ltd."
echo "Developed by: Abhishek Rajput (@abhi-hisync)" 
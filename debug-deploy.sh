#!/bin/bash

# HISYNC AI - Debug Deployment Script
# This script helps diagnose virtual environment issues

echo "🔍 HISYNC AI - Debug Mode"
echo "========================="
echo "Diagnosing virtual environment issues..."
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

echo "🐍 Python Information:"
echo "   Command: $PYTHON_CMD"
echo "   Version: $PYTHON_VERSION"
echo "   Location: $(which $PYTHON_CMD)"
echo ""

echo "🔍 Checking Python modules:"
echo -n "   venv module: "
if $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
    echo "✅ Available"
    VENV_AVAILABLE=true
else
    echo "❌ Not available"
    VENV_AVAILABLE=false
fi

echo -n "   virtualenv module: "
if $PYTHON_CMD -m virtualenv --help >/dev/null 2>&1; then
    echo "✅ Available"
    VIRTUALENV_AVAILABLE=true
else
    echo "❌ Not available"
    VIRTUALENV_AVAILABLE=false
fi

echo -n "   pip module: "
if $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo "✅ Available"
    PIP_AVAILABLE=true
else
    echo "❌ Not available"
    PIP_AVAILABLE=false
fi

echo ""
echo "📁 Current directory: $(pwd)"
echo "📋 Directory contents:"
ls -la

echo ""
echo "🔍 Virtual environment test:"

# Clean up any existing venv
if [ -d "venv" ]; then
    echo "🧹 Removing existing venv directory..."
    rm -rf venv
fi

# Try to create virtual environment
if [ "$VENV_AVAILABLE" = true ]; then
    echo "🧪 Testing venv module..."
    if $PYTHON_CMD -m venv test-venv; then
        echo "✅ venv creation successful"
        if [ -f "test-venv/bin/activate" ]; then
            echo "✅ Activation script exists"
            echo "📁 test-venv structure:"
            ls -la test-venv/
            echo "📁 test-venv/bin contents:"
            ls -la test-venv/bin/
            rm -rf test-venv
        else
            echo "❌ Activation script missing"
            echo "📁 test-venv structure:"
            ls -la test-venv/
            rm -rf test-venv
        fi
    else
        echo "❌ venv creation failed"
    fi
elif [ "$PIP_AVAILABLE" = true ]; then
    echo "🧪 Installing and testing virtualenv..."
    if $PYTHON_CMD -m pip install --user virtualenv --quiet; then
        echo "✅ virtualenv installed"
        if $PYTHON_CMD -m virtualenv test-venv; then
            echo "✅ virtualenv creation successful"
            if [ -f "test-venv/bin/activate" ]; then
                echo "✅ Activation script exists"
                rm -rf test-venv
            else
                echo "❌ Activation script missing"
                echo "📁 test-venv structure:"
                ls -la test-venv/
                rm -rf test-venv
            fi
        else
            echo "❌ virtualenv creation failed"
        fi
    else
        echo "❌ virtualenv installation failed"
    fi
else
    echo "❌ No virtual environment method available"
fi

echo ""
echo "🔧 System Information:"
echo "   OS: $(uname -a)"
echo "   User: $(whoami)"
echo "   Home: $HOME"
echo "   PATH: $PATH"

echo ""
echo "📋 Python packages (user):"
$PYTHON_CMD -m pip list --user 2>/dev/null || echo "No user packages or pip not available"

echo ""
echo "💡 Recommendations:"
if [ "$VENV_AVAILABLE" = false ] && [ "$VIRTUALENV_AVAILABLE" = false ]; then
    echo "   ❌ No virtual environment support found"
    echo "   💡 Ask your hosting provider to install python3-venv package"
    echo "   💡 Or try: $PYTHON_CMD -m pip install --user virtualenv"
elif [ "$VENV_AVAILABLE" = true ]; then
    echo "   ✅ venv module available - should work"
else
    echo "   ✅ virtualenv available - should work"
fi

echo ""
echo "🎯 Next Steps:"
echo "   1. If venv/virtualenv is available, run: ./deploy-no-sudo.sh"
echo "   2. If not available, contact your hosting provider"
echo "   3. Share this debug output for further assistance"

echo ""
echo "�� Debug complete!" 
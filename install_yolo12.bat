@echo off
REM HISYNC AI - YOLO12 Installation Script for Windows
REM © 2025 Hire Synchronisation Pvt. Ltd.
REM Developer: Abhishek Rajput (@abhi-hisync)

echo.
echo 🔥 HISYNC AI - YOLO12 Installation Script
echo ==========================================
echo Installing YOLO12 attention-centric object detection...
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Check pip version
echo 📋 Checking pip version...
pip --version
if %errorlevel% neq 0 (
    echo ❌ pip not found. Please install pip first.
    pause
    exit /b 1
)

echo.
echo 📥 Installing YOLO12 dependencies...

REM Install PyTorch (CPU version)
echo 🔥 Installing PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo ❌ Failed to install PyTorch
    pause
    exit /b 1
)

REM Install Ultralytics (includes YOLO12)
echo 🎯 Installing Ultralytics YOLO12...
pip install "ultralytics>=8.3.170"
if %errorlevel% neq 0 (
    echo ❌ Failed to install Ultralytics
    pause
    exit /b 1
)

REM Install OpenCV
echo 📸 Installing OpenCV...
pip install opencv-python-headless
if %errorlevel% neq 0 (
    echo ❌ Failed to install OpenCV
    pause
    exit /b 1
)

REM Install additional dependencies
echo 🔧 Installing additional dependencies...
pip install "pillow>=10.1.0"
pip install "numpy>=1.24.3"

REM Verify installation
echo.
echo ✅ Verifying YOLO12 installation...
python -c "try: from ultralytics import YOLO; import torch; print('✅ YOLO12 imported successfully!'); print(f'✅ PyTorch version: {torch.__version__}'); model = YOLO('yolo12n.pt'); print('✅ YOLO12 nano model loaded successfully!'); print('🚀 YOLO12 installation completed successfully!'); except ImportError as e: print(f'❌ Installation failed: {e}'); exit(1)"

if %errorlevel% neq 0 (
    echo ❌ YOLO12 verification failed
    pause
    exit /b 1
)

echo.
echo 🎉 YOLO12 Installation Complete!
echo =================================
echo ✅ YOLO12 attention-centric object detection is ready!
echo ✅ You can now use YOLO12 for:
echo    - Object Detection with Area Attention
echo    - Instance Segmentation with R-ELAN
echo    - Image Classification with FlashAttention
echo    - Pose Estimation
echo    - Oriented Object Detection (OBB)
echo.
echo 🚀 Start the HISYNC AI server:
echo    python main.py
echo.
echo 📚 Documentation: /docs endpoint
echo 🏢 Support: support@hisync.in
echo.
echo © 2025 Hire Synchronisation Pvt. Ltd.
echo.
pause

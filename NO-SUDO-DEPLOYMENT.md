# 🚀 HISYNC AI - No-Sudo Deployment Guide

**Deploy HISYNC AI without sudo/root access - Perfect for shared hosting!**

---

**© 2024 HISYNC Technologies. All rights reserved.**  
**Support**: support@hisync.in | **Website**: https://hisync.in

---

## ✅ **No Sudo Required!**

Bhai, **bilkul sudo password ki zarurat nahi hai!** Ye script normal user access se kaam karta hai.

### 🎯 **One-Command Deployment:**

```bash
# Upload your HISYNC AI files to server
# SSH into server and run:

chmod +x deploy-no-sudo.sh
./deploy-no-sudo.sh
```

**That's it!** Script automatically:
- ✅ Detects available Python version (3.8+)
- ✅ Creates virtual environment without sudo
- ✅ Installs all dependencies locally
- ✅ Creates process manager (no systemd needed)
- ✅ Starts HISYNC AI service
- ✅ Tests deployment
- ✅ Provides nginx config for Forge

---

## 📋 **Requirements (Minimal):**

- ✅ **Python 3.8+** (usually pre-installed on servers)
- ✅ **Basic user account** (no sudo needed)
- ✅ **Internet access** (for pip packages)
- ✅ **Laravel Forge account** (for domain/SSL)

### Check if you have Python:
```bash
python3 --version
# Should show: Python 3.8+ or higher
```

If no Python 3, ask your hosting provider to install it (they usually do this without issues).

---

## 🔧 **How It Works (No Sudo Magic):**

### 1. **Auto Python Detection:**
```bash
# Script checks for these in order:
python3.12 → python3.11 → python3.10 → python3.9 → python3.8 → python3
```

### 2. **Local Virtual Environment:**
```bash
# Creates venv without system packages
python3 -m venv venv
# OR if venv not available:
python3 -m pip install --user virtualenv
```

### 3. **User-Level Installation:**
```bash
# All packages installed in venv (no system changes)
pip install -r requirements.txt
```

### 4. **Process Management (No systemd):**
```bash
# Simple bash-based process manager
./start-hisync.sh start    # Start HISYNC AI
./start-hisync.sh stop     # Stop HISYNC AI
./start-hisync.sh status   # Check status
./start-hisync.sh logs     # View logs
```

---

## 📁 **Files Created (All User-Level):**

```
your-project/
├── 🔥 HISYNC AI files (your code)
├── 🐍 venv/ (virtual environment)
├── 🚀 start-hisync.sh (process manager)
├── 🔄 auto-restart.sh (auto-restart script)
├── 📋 nginx-config.txt (for Laravel Forge)
├── 📝 hisync-ai.log (application logs)
├── 🆔 hisync-ai.pid (process ID)
└── ⚙️ .env (configuration)
```

**No system files touched!** Everything in your user directory.

---

## 🌐 **Laravel Forge Setup:**

### Step 1: Create Site in Forge
- **Domain**: `ai.hisync.in`
- **Project Type**: Static HTML
- **Directory**: `/home/forge/ai.hisync.in`

### Step 2: Upload HISYNC AI Code
```bash
# Option A: Git (Recommended)
git clone your-repo.git /home/forge/ai.hisync.in

# Option B: Upload via Forge dashboard
# Upload all files to /home/forge/ai.hisync.in
```

### Step 3: Run Deployment
```bash
ssh forge@your-server
cd /home/forge/ai.hisync.in
chmod +x deploy-no-sudo.sh
./deploy-no-sudo.sh
```

### Step 4: Configure Nginx in Forge
Copy content from `nginx-config.txt` to your site's nginx configuration in Laravel Forge dashboard.

### Step 5: Enable SSL
Enable SSL certificate in Laravel Forge dashboard for `ai.hisync.in`.

---

## 🎯 **Management Commands:**

```bash
# Check if HISYNC AI is running
./start-hisync.sh status

# Start HISYNC AI
./start-hisync.sh start

# Stop HISYNC AI
./start-hisync.sh stop

# Restart HISYNC AI
./start-hisync.sh restart

# View real-time logs
./start-hisync.sh logs

# Check log file directly
tail -f hisync-ai.log
```

---

## 🔄 **Auto-Restart Setup (Optional):**

```bash
# Add to crontab for auto-restart
crontab -e

# Add this line:
*/5 * * * * cd /home/forge/ai.hisync.in && ./auto-restart.sh
```

This checks every 5 minutes and restarts HISYNC AI if it's down.

---

## 🚨 **Troubleshooting:**

### Issue 1: Python Not Found
```bash
# Check available Python versions
ls /usr/bin/python*

# Try specific version
python3.9 --version
```

### Issue 2: Virtual Environment Fails
```bash
# Install virtualenv locally
python3 -m pip install --user virtualenv

# Create venv with virtualenv
python3 -m virtualenv venv
```

### Issue 3: Permission Denied
```bash
# Make scripts executable
chmod +x deploy-no-sudo.sh
chmod +x start-hisync.sh
chmod +x auto-restart.sh
```

### Issue 4: Port Already in Use
```bash
# Check what's using port 8000
netstat -tlnp | grep :8000

# Kill existing process
./start-hisync.sh stop
```

### Issue 5: Dependencies Installation Fails
```bash
# Clear pip cache and retry
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

---

## 📊 **Testing Your Deployment:**

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test company info
curl http://localhost:8000/company

# Test classification (with image)
curl -X POST http://localhost:8000/classify \
     -F "image=@test.jpg" \
     -F "expected_label=cat" \
     -F "confidence_threshold=0.8"
```

---

## 🎉 **Success Checklist:**

- [ ] Python 3.8+ available
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] HISYNC AI process running
- [ ] Health check responds
- [ ] Nginx configured in Forge
- [ ] SSL certificate enabled
- [ ] Domain pointing to server

---

## 💡 **Pro Tips:**

### 1. **Memory Management:**
```bash
# Check memory usage
free -h

# Monitor HISYNC AI memory
./start-hisync.sh status
```

### 2. **Log Management:**
```bash
# Rotate logs (prevent large files)
echo "" > hisync-ai.log  # Clear logs

# Monitor logs in real-time
tail -f hisync-ai.log | grep -E "(ERROR|WARNING|INFO)"
```

### 3. **Performance Monitoring:**
```bash
# Check CPU usage
top -p $(cat hisync-ai.pid)

# Network connections
netstat -tlnp | grep :8000
```

---

## 🔥 **Advantages of No-Sudo Deployment:**

✅ **Security**: No system-level changes  
✅ **Portability**: Works on any hosting  
✅ **Isolation**: Everything in user space  
✅ **Simplicity**: No complex setup  
✅ **Maintenance**: Easy to manage  
✅ **Cost-Effective**: Works on shared hosting  

---

## 📞 **Support:**

- **Email**: support@hisync.in
- **Website**: https://hisync.in
- **Documentation**: This guide
- **24/7 Support**: Available for enterprise clients

---

**🔥 Bhai, ab tumhara HISYNC AI bina sudo ke chal jayega!**

**No sudo, no problem! 🚀**

**© 2024 HISYNC Technologies. All rights reserved.** 
# 🚀 HISYNC AI - Laravel Forge Deployment Guide

**Deploy HISYNC AI Image Classification API on ai.hisync.in**

---

**© 2024 HISYNC Technologies. All rights reserved.**  
**Support**: support@hisync.in | **Website**: https://hisync.in

---

## 🎯 Quick Deployment (Automated)

### Step 1: Upload Code to Forge Server

1. **Upload your HISYNC AI code** to Laravel Forge server:
   ```bash
   # Option A: Git Repository (Recommended)
   git clone https://github.com/yourusername/hisync-ai.git /home/forge/ai.hisync.in
   
   # Option B: Direct Upload via Forge Dashboard
   # Upload all files to /home/forge/ai.hisync.in
   ```

2. **SSH into your Forge server**:
   ```bash
   ssh forge@your-server-ip
   cd /home/forge/ai.hisync.in
   ```

### Step 2: Run Automated Deployment

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run HISYNC AI deployment
./deploy.sh
```

**That's it!** The script will automatically:
- ✅ Install Python dependencies
- ✅ Create virtual environment
- ✅ Configure systemd service
- ✅ Set up process monitoring
- ✅ Start HISYNC AI service
- ✅ Test the deployment

---

## 🔧 Manual Deployment (Step by Step)

### Prerequisites Checklist

- [ ] Laravel Forge account active
- [ ] Server created (4GB+ RAM recommended)
- [ ] Domain `ai.hisync.in` pointed to server
- [ ] SSL certificate enabled in Forge
- [ ] SSH access to server

### Step 1: Server Preparation

```bash
# SSH into your Forge server
ssh forge@your-server-ip

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-venv python3-pip -y
sudo apt install python3-dev python3-setuptools -y
sudo apt install libhdf5-dev pkg-config -y
sudo apt install supervisor -y
```

### Step 2: Deploy HISYNC AI Code

```bash
# Navigate to site directory
cd /home/forge/ai.hisync.in

# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install HISYNC AI dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Create environment file
nano .env
```

Add this configuration:

```bash
# HISYNC AI Configuration
APP_NAME="HISYNC AI - Image Classification API"
COMPANY="HISYNC Technologies"
DEBUG=False
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-super-secure-secret-key-here

# Performance Settings
MAX_WORKERS=4
TIMEOUT=60

# Support Information
SUPPORT_EMAIL=support@hisync.in
COMPANY_WEBSITE=https://hisync.in
```

### Step 4: Create System Service

```bash
# Create systemd service
sudo nano /etc/systemd/system/hisync-ai.service
```

Add this configuration:

```ini
[Unit]
Description=HISYNC AI - Image Classification API
After=network.target

[Service]
Type=simple
User=forge
Group=forge
WorkingDirectory=/home/forge/ai.hisync.in
Environment=PATH=/home/forge/ai.hisync.in/venv/bin
ExecStart=/home/forge/ai.hisync.in/venv/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Step 5: Configure Process Monitoring

```bash
# Create supervisor configuration
sudo nano /etc/supervisor/conf.d/hisync-ai.conf
```

Add this configuration:

```ini
[program:hisync-ai]
command=/home/forge/ai.hisync.in/venv/bin/python main.py
directory=/home/forge/ai.hisync.in
user=forge
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/hisync-ai.log
environment=PATH="/home/forge/ai.hisync.in/venv/bin"
```

### Step 6: Configure Nginx (In Forge Dashboard)

Go to your site in Laravel Forge and update the **Nginx Configuration**:

```nginx
server {
    listen 80;
    listen 443 ssl http2;
    server_name ai.hisync.in;
    root /home/forge/ai.hisync.in;

    # SSL Configuration (managed by Forge)
    ssl_certificate /etc/nginx/ssl/ai.hisync.in/server.crt;
    ssl_certificate_key /etc/nginx/ssl/ai.hisync.in/server.key;

    # HISYNC AI API Proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Handle large file uploads for HISYNC AI
        client_max_body_size 10M;
        
        # Timeouts for AI processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    # Security headers for HISYNC AI
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
}
```

### Step 7: Start Services

```bash
# Enable and start systemd service
sudo systemctl daemon-reload
sudo systemctl enable hisync-ai

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start hisync-ai

# Check status
sudo supervisorctl status hisync-ai
```

### Step 8: Test Deployment

```bash
# Test local connection
curl http://localhost:8000/health

# Test domain connection
curl https://ai.hisync.in/health

# Test HISYNC AI classification
curl -X POST "https://ai.hisync.in/classify" \
     -F "image=@test_image.jpg" \
     -F "expected_label=cat" \
     -F "confidence_threshold=0.8"
```

---

## 📊 Management Commands

### Service Management
```bash
# Check service status
sudo supervisorctl status hisync-ai

# Start service
sudo supervisorctl start hisync-ai

# Stop service
sudo supervisorctl stop hisync-ai

# Restart service
sudo supervisorctl restart hisync-ai

# Reload configuration
sudo supervisorctl reread && sudo supervisorctl update
```

### Monitoring
```bash
# View real-time logs
tail -f /var/log/hisync-ai.log

# Check system resources
htop

# Monitor API health
watch -n 5 'curl -s https://ai.hisync.in/health | jq'

# Check service status
systemctl status hisync-ai
```

### Maintenance
```bash
# Update HISYNC AI code
cd /home/forge/ai.hisync.in
git pull origin main
sudo supervisorctl restart hisync-ai

# Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt
sudo supervisorctl restart hisync-ai

# Clean logs
sudo truncate -s 0 /var/log/hisync-ai.log
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. Service Won't Start
```bash
# Check logs
journalctl -u hisync-ai -f

# Check supervisor logs
tail -f /var/log/hisync-ai.log

# Verify Python path
which python3.9
ls -la /home/forge/ai.hisync.in/venv/bin/python
```

#### 2. Port Already in Use
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
sudo kill -9 PID

# Restart service
sudo supervisorctl restart hisync-ai
```

#### 3. TensorFlow Installation Issues
```bash
# Reinstall with no cache
source venv/bin/activate
pip uninstall tensorflow
pip install --no-cache-dir tensorflow==2.15.0

# Check installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### 4. Memory Issues
```bash
# Check memory usage
free -h

# Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 5. SSL Certificate Issues
```bash
# Renew certificate in Forge dashboard
# Or manually:
sudo certbot renew

# Test SSL
curl -I https://ai.hisync.in
```

#### 6. Permission Issues
```bash
# Fix ownership
sudo chown -R forge:forge /home/forge/ai.hisync.in

# Fix permissions
chmod +x /home/forge/ai.hisync.in/main.py
chmod +x /home/forge/ai.hisync.in/deploy.sh
```

---

## 🔧 Performance Optimization

### 1. Multi-Worker Setup
Update `main.py` for production:

```python
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=4,  # Use 4 workers for better performance
        reload=False,  # Disable in production
        log_level="info"
    )
```

### 2. Redis Caching (Optional)
```bash
# Install Redis
sudo apt install redis-server -y

# Add to requirements.txt
echo "redis==4.5.1" >> requirements.txt
pip install redis==4.5.1
```

### 3. Database Logging (Optional)
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Create database for logging
sudo -u postgres createdb hisync_ai_logs
```

---

## 📈 Monitoring & Analytics

### 1. Set up Log Rotation
```bash
sudo nano /etc/logrotate.d/hisync-ai
```

Add:
```
/var/log/hisync-ai.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 forge forge
    postrotate
        supervisorctl restart hisync-ai
    endscript
}
```

### 2. Health Monitoring Script
```bash
# Create monitoring script
nano /home/forge/monitor-hisync.sh
```

Add:
```bash
#!/bin/bash
HEALTH_URL="https://ai.hisync.in/health"
WEBHOOK_URL="your-slack-webhook-url"

if ! curl -s -f $HEALTH_URL > /dev/null; then
    curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"🚨 HISYNC AI is DOWN! Please check ai.hisync.in"}' \
    $WEBHOOK_URL
fi
```

### 3. Set up Cron for Monitoring
```bash
# Add to crontab
crontab -e

# Add this line for 5-minute monitoring
*/5 * * * * /home/forge/monitor-hisync.sh
```

---

## 🎉 Deployment Complete!

### ✅ Verification Checklist

- [ ] HISYNC AI service is running
- [ ] Health check responds: `https://ai.hisync.in/health`
- [ ] API documentation accessible: `https://ai.hisync.in/docs`
- [ ] Image classification working
- [ ] SSL certificate active
- [ ] Process monitoring active
- [ ] Logs are being written

### 🌐 Access Points

- **API Base**: https://ai.hisync.in
- **Documentation**: https://ai.hisync.in/docs
- **Health Check**: https://ai.hisync.in/health
- **Company Info**: https://ai.hisync.in/company

### 📞 Support

- **Email**: support@hisync.in
- **Website**: https://hisync.in
- **Documentation**: This guide
- **24/7 Support**: Available for enterprise clients

---

**🔥 HISYNC AI is now live and ready for enterprise use!**

**© 2024 HISYNC Technologies. All rights reserved.** 
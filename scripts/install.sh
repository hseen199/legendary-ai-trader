#!/bin/bash
#
# Legendary AI Trading Platform - Installation Script
# سكريبت تثبيت منصة التداول الأسطورية
#
# الاستخدام: bash install.sh
#

set -e

# الألوان للطباعة
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# الشعار
print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║   🚀 LEGENDARY AI TRADING PLATFORM V3.0                      ║"
    echo "║   منصة التداول الأسطورية بالذكاء الاصطناعي                    ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# طباعة رسالة
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# التحقق من المتطلبات
check_requirements() {
    log_info "التحقق من المتطلبات..."
    
    # التحقق من أننا على Ubuntu
    if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
        log_warning "هذا السكريبت مصمم لـ Ubuntu. قد تحتاج لتعديلات على أنظمة أخرى."
    fi
    
    # التحقق من صلاحيات sudo
    if ! sudo -v; then
        log_error "تحتاج صلاحيات sudo لتشغيل هذا السكريبت"
        exit 1
    fi
    
    log_success "تم التحقق من المتطلبات"
}

# تحديث النظام
update_system() {
    log_info "تحديث النظام..."
    sudo apt-get update -y
    sudo apt-get upgrade -y
    log_success "تم تحديث النظام"
}

# تثبيت الأدوات الأساسية
install_basics() {
    log_info "تثبيت الأدوات الأساسية..."
    sudo apt-get install -y \
        curl \
        wget \
        git \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        unzip \
        htop \
        vim \
        nano
    log_success "تم تثبيت الأدوات الأساسية"
}

# تثبيت Python 3.11
install_python() {
    log_info "تثبيت Python 3.11..."
    
    # إضافة مستودع deadsnakes
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    
    # تثبيت Python 3.11
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
    
    # جعل Python 3.11 الافتراضي
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    
    log_success "تم تثبيت Python $(python3.11 --version)"
}

# تثبيت Node.js
install_nodejs() {
    log_info "تثبيت Node.js 20..."
    
    # إضافة مستودع NodeSource
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # تثبيت pnpm
    sudo npm install -g pnpm
    
    log_success "تم تثبيت Node.js $(node --version) و pnpm $(pnpm --version)"
}

# تثبيت PostgreSQL
install_postgresql() {
    log_info "تثبيت PostgreSQL..."
    
    sudo apt-get install -y postgresql postgresql-contrib
    
    # بدء الخدمة
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    
    log_success "تم تثبيت PostgreSQL"
}

# تثبيت Redis
install_redis() {
    log_info "تثبيت Redis..."
    
    sudo apt-get install -y redis-server
    
    # بدء الخدمة
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
    
    log_success "تم تثبيت Redis"
}

# تثبيت Nginx
install_nginx() {
    log_info "تثبيت Nginx..."
    
    sudo apt-get install -y nginx
    
    # بدء الخدمة
    sudo systemctl start nginx
    sudo systemctl enable nginx
    
    log_success "تم تثبيت Nginx"
}

# تثبيت Docker (اختياري)
install_docker() {
    log_info "تثبيت Docker..."
    
    # إضافة مفتاح GPG
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # إضافة المستودع
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # تثبيت Docker
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # إضافة المستخدم لمجموعة docker
    sudo usermod -aG docker $USER
    
    log_success "تم تثبيت Docker"
}

# إعداد قاعدة البيانات
setup_database() {
    log_info "إعداد قاعدة البيانات..."
    
    # إنشاء المستخدم وقاعدة البيانات
    sudo -u postgres psql -c "CREATE USER legendary WITH PASSWORD 'legendary_secure_password';" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE DATABASE legendary_platform OWNER legendary;" 2>/dev/null || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE legendary_platform TO legendary;" 2>/dev/null || true
    
    log_success "تم إعداد قاعدة البيانات"
    log_warning "تذكر تغيير كلمة المرور في ملف .env"
}

# إعداد المشروع
setup_project() {
    log_info "إعداد المشروع..."
    
    # الانتقال لمجلد المشروع
    PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
    cd $PROJECT_DIR
    
    # إنشاء البيئة الافتراضية
    log_info "إنشاء البيئة الافتراضية..."
    python3.11 -m venv venv
    source venv/bin/activate
    
    # تحديث pip
    pip install --upgrade pip
    
    # تثبيت متطلبات Backend
    log_info "تثبيت متطلبات Backend..."
    pip install -r backend/requirements.txt
    
    # تثبيت متطلبات Bot
    log_info "تثبيت متطلبات Bot..."
    pip install -r bot/requirements.txt
    
    # تثبيت متطلبات Frontend
    log_info "تثبيت متطلبات Frontend..."
    cd frontend
    pnpm install
    cd ..
    
    # نسخ ملف البيئة
    if [ ! -f .env ]; then
        cp .env.example .env
        log_warning "تم إنشاء ملف .env - يرجى تعديله بالإعدادات الصحيحة"
    fi
    
    log_success "تم إعداد المشروع"
}

# إعداد Nginx
setup_nginx() {
    log_info "إعداد Nginx..."
    
    # إنشاء ملف التكوين
    sudo tee /etc/nginx/sites-available/legendary-platform > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
EOF

    # تفعيل الموقع
    sudo ln -sf /etc/nginx/sites-available/legendary-platform /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # اختبار التكوين
    sudo nginx -t
    
    # إعادة تشغيل Nginx
    sudo systemctl restart nginx
    
    log_success "تم إعداد Nginx"
}

# إنشاء خدمات systemd
create_services() {
    log_info "إنشاء خدمات systemd..."
    
    PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
    
    # خدمة Backend
    sudo tee /etc/systemd/system/legendary-backend.service > /dev/null << EOF
[Unit]
Description=Legendary AI Trading Platform - Backend
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/backend
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # خدمة Bot
    sudo tee /etc/systemd/system/legendary-bot.service > /dev/null << EOF
[Unit]
Description=Legendary AI Trading Platform - Trading Bot
After=network.target legendary-backend.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/bot
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python main_integrated.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

    # خدمة Frontend
    sudo tee /etc/systemd/system/legendary-frontend.service > /dev/null << EOF
[Unit]
Description=Legendary AI Trading Platform - Frontend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/frontend
ExecStart=/usr/bin/pnpm start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # إعادة تحميل systemd
    sudo systemctl daemon-reload
    
    log_success "تم إنشاء خدمات systemd"
}

# بدء الخدمات
start_services() {
    log_info "بدء الخدمات..."
    
    sudo systemctl enable legendary-backend
    sudo systemctl enable legendary-bot
    sudo systemctl enable legendary-frontend
    
    sudo systemctl start legendary-backend
    sudo systemctl start legendary-frontend
    
    log_success "تم بدء الخدمات"
    log_warning "البوت لم يبدأ تلقائياً - شغّله يدوياً بعد التأكد من الإعدادات"
}

# الرسالة النهائية
print_final_message() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║   ✅ تم التثبيت بنجاح!                                        ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}الخطوات التالية:${NC}"
    echo ""
    echo "1. عدّل ملف .env بالإعدادات الصحيحة:"
    echo "   nano .env"
    echo ""
    echo "2. أضف مفاتيح Binance API"
    echo ""
    echo "3. شغّل البوت:"
    echo "   sudo systemctl start legendary-bot"
    echo ""
    echo "4. افتح المتصفح على:"
    echo "   http://YOUR_SERVER_IP"
    echo ""
    echo -e "${YELLOW}أوامر مفيدة:${NC}"
    echo "  sudo systemctl status legendary-backend"
    echo "  sudo systemctl status legendary-bot"
    echo "  sudo journalctl -u legendary-bot -f"
    echo ""
}

# التنفيذ الرئيسي
main() {
    print_banner
    
    check_requirements
    update_system
    install_basics
    install_python
    install_nodejs
    install_postgresql
    install_redis
    install_nginx
    install_docker
    setup_database
    setup_project
    setup_nginx
    create_services
    start_services
    
    print_final_message
}

# تشغيل السكريبت
main "$@"

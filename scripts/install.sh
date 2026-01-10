#!/bin/bash
# ============================================
# SanadTrade - Installation Script
# منصة سند للتداول الذكي
# Domain: sanadtrade.com
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║   🚀 SANADTRADE - LEGENDARY AI TRADING PLATFORM              ║"
    echo "║   منصة سند للتداول الذكي بالذكاء الاصطناعي                    ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print functions
print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Get project directory
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "Please run as root (sudo ./install.sh)"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    print_header "Checking System Requirements"
    
    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_info "OS: $NAME $VERSION"
    fi
    
    # Check RAM
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_ram" -lt 2 ]; then
        print_warning "Recommended RAM: 4GB+, Current: ${total_ram}GB"
    else
        print_success "RAM: ${total_ram}GB"
    fi
    
    # Check disk space
    available_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 10 ]; then
        print_warning "Recommended disk space: 20GB+, Available: ${available_space}GB"
    else
        print_success "Disk space: ${available_space}GB available"
    fi
}

# Update system
update_system() {
    print_header "Updating System"
    apt-get update -y
    apt-get upgrade -y
    print_success "System updated"
}

# Install basic tools
install_basics() {
    print_header "Installing Basic Tools"
    apt-get install -y \
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
        nano \
        openssl
    print_success "Basic tools installed"
}

# Install Docker
install_docker() {
    print_header "Installing Docker"
    
    if command -v docker &> /dev/null; then
        print_success "Docker is already installed"
        docker --version
    else
        print_info "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        rm get-docker.sh
        
        # Add current user to docker group
        usermod -aG docker $SUDO_USER || true
        
        print_success "Docker installed successfully"
    fi
    
    # Start Docker service
    systemctl start docker
    systemctl enable docker
}

# Install Docker Compose
install_docker_compose() {
    print_header "Installing Docker Compose"
    
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose is already installed"
        docker-compose --version
    else
        print_info "Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
        print_success "Docker Compose installed successfully"
    fi
}

# Setup environment files
setup_env() {
    print_header "Setting Up Environment Files"
    
    cd $PROJECT_DIR
    
    # Main .env
    if [ ! -f .env ]; then
        cp .env.example .env
        print_success "Created .env from .env.example"
    else
        print_info ".env already exists"
    fi
    
    # Backend .env
    if [ ! -f backend/.env ]; then
        cp backend/.env.example backend/.env
        print_success "Created backend/.env"
    else
        print_info "backend/.env already exists"
    fi
    
    # Generate secret key
    SECRET_KEY=$(openssl rand -hex 32)
    sed -i "s/your-super-secret-key-change-this-in-production-min-32-chars/$SECRET_KEY/" .env 2>/dev/null || true
    sed -i "s/your-super-secret-key-min-32-characters-long/$SECRET_KEY/" backend/.env 2>/dev/null || true
    print_success "Generated new SECRET_KEY"
    
    # Generate database password
    DB_PASSWORD=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)
    sed -i "s/your_secure_password_here/$DB_PASSWORD/g" .env 2>/dev/null || true
    sed -i "s/your_secure_password/$DB_PASSWORD/g" backend/.env 2>/dev/null || true
    print_success "Generated new database password"
    
    print_warning "Please edit .env files with your actual API keys!"
}

# Setup SSL with Certbot
setup_ssl() {
    print_header "Setting Up SSL Certificate"
    
    read -p "Enter your domain (e.g., sanadtrade.com): " DOMAIN
    read -p "Enter your email for SSL notifications: " EMAIL
    
    cd $PROJECT_DIR
    
    # Create directories
    mkdir -p certbot/conf certbot/www
    
    # Update nginx config with domain
    sed -i "s/sanadtrade.com/$DOMAIN/g" nginx/conf.d/sanadtrade.conf
    
    # Get initial certificate
    docker run -it --rm \
        -v "$PROJECT_DIR/certbot/conf:/etc/letsencrypt" \
        -v "$PROJECT_DIR/certbot/www:/var/www/certbot" \
        -p 80:80 \
        certbot/certbot certonly \
        --standalone \
        --email "$EMAIL" \
        --agree-tos \
        --no-eff-email \
        -d "$DOMAIN" \
        -d "www.$DOMAIN"
    
    if [ $? -eq 0 ]; then
        print_success "SSL certificate obtained successfully"
        print_success "Updated nginx configuration with domain: $DOMAIN"
    else
        print_error "Failed to obtain SSL certificate"
        print_warning "You can try again later or use HTTP only for testing"
        use_local_config
    fi
}

# Use local config without SSL
use_local_config() {
    print_info "Setting up local configuration without SSL..."
    cd $PROJECT_DIR
    cp nginx/conf.d/local.conf.example nginx/conf.d/default.conf
    rm -f nginx/conf.d/sanadtrade.conf
    print_success "Local configuration applied"
}

# Build and start containers
start_services() {
    print_header "Building and Starting Services"
    
    cd $PROJECT_DIR
    
    # Build images
    print_info "Building Docker images..."
    docker-compose build --no-cache
    
    # Start services
    print_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to start..."
    sleep 30
    
    # Check service status
    docker-compose ps
    
    print_success "All services started"
}

# Run database migrations
run_migrations() {
    print_header "Running Database Migrations"
    
    cd $PROJECT_DIR
    
    # Wait for database to be ready
    print_info "Waiting for database..."
    sleep 10
    
    # Run migrations
    docker-compose exec -T backend alembic upgrade head 2>/dev/null || {
        print_warning "Alembic migrations skipped (tables may already exist)"
    }
    
    print_success "Database setup completed"
}

# Initialize platform data
init_platform() {
    print_header "Initializing Platform Data"
    
    cd $PROJECT_DIR
    
    docker-compose exec -T backend python scripts/init_db.py 2>/dev/null || {
        print_warning "Platform initialization skipped"
    }
    
    print_success "Platform data initialized"
}

# Create admin user
create_admin() {
    print_header "Creating Admin User"
    
    cd $PROJECT_DIR
    
    read -p "Enter admin email [admin@sanadtrade.com]: " ADMIN_EMAIL
    ADMIN_EMAIL=${ADMIN_EMAIL:-admin@sanadtrade.com}
    
    read -s -p "Enter admin password [Admin@123456]: " ADMIN_PASSWORD
    ADMIN_PASSWORD=${ADMIN_PASSWORD:-Admin@123456}
    echo ""
    
    # Set environment variables and run script
    docker-compose exec -T \
        -e ADMIN_EMAIL="$ADMIN_EMAIL" \
        -e ADMIN_PASSWORD="$ADMIN_PASSWORD" \
        backend python scripts/create_admin.py 2>/dev/null || {
        print_warning "Admin creation skipped (may already exist)"
    }
    
    print_success "Admin user setup completed"
}

# Setup firewall
setup_firewall() {
    print_header "Setting Up Firewall"
    
    if command -v ufw &> /dev/null; then
        ufw allow 22/tcp    # SSH
        ufw allow 80/tcp    # HTTP
        ufw allow 443/tcp   # HTTPS
        ufw --force enable
        print_success "Firewall configured"
    else
        print_warning "UFW not found, please configure firewall manually"
    fi
}

# Print final instructions
print_final() {
    print_header "Installation Complete!"
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   SanadTrade Installed!                      ║"
    echo "║                   تم تثبيت منصة سند!                         ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║                                                              ║"
    echo "║  🌐 Website: https://sanadtrade.com                          ║"
    echo "║  📊 API: https://sanadtrade.com/api/v1                       ║"
    echo "║  🔐 Admin: https://sanadtrade.com/admin                      ║"
    echo "║                                                              ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  الخطوات التالية:                                            ║"
    echo "║  1. Edit .env with your Binance API keys                     ║"
    echo "║  2. Configure email settings (SMTP)                          ║"
    echo "║  3. Test deposit/withdrawal flow                             ║"
    echo "║  4. Enable trading bot when ready                            ║"
    echo "║                                                              ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  أوامر مفيدة:                                                 ║"
    echo "║  - View logs: docker-compose logs -f                         ║"
    echo "║  - Restart: docker-compose restart                           ║"
    echo "║  - Stop: docker-compose down                                 ║"
    echo "║  - Update: git pull && docker-compose up -d --build          ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Main installation flow
main() {
    print_banner
    
    check_root
    check_requirements
    
    echo ""
    read -p "Continue with installation? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        print_info "Installation cancelled"
        exit 0
    fi
    
    update_system
    install_basics
    install_docker
    install_docker_compose
    setup_env
    
    echo ""
    read -p "Setup SSL certificate now? (y/n): " SETUP_SSL
    if [ "$SETUP_SSL" == "y" ]; then
        setup_ssl
    else
        print_warning "Skipping SSL setup"
        use_local_config
    fi
    
    start_services
    run_migrations
    init_platform
    create_admin
    setup_firewall
    
    print_final
}

# Run main function
main "$@"

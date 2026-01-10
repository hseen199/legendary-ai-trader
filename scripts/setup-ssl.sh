#!/bin/bash
# ============================================
# SanadTrade - SSL Setup Script
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Get project directory
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (sudo ./setup-ssl.sh)"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              SanadTrade SSL Setup                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get domain and email
read -p "Enter your domain (e.g., sanadtrade.com): " DOMAIN
read -p "Enter your email for SSL notifications: " EMAIL

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    print_error "Domain and email are required"
    exit 1
fi

cd $PROJECT_DIR

# Stop nginx if running
print_info "Stopping nginx..."
docker-compose stop nginx 2>/dev/null || true

# Create directories
mkdir -p certbot/conf certbot/www

# Get certificate
print_info "Obtaining SSL certificate for $DOMAIN..."
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
    print_success "SSL certificate obtained successfully!"
    
    # Update nginx config
    print_info "Updating nginx configuration..."
    
    # Remove local config if exists
    rm -f nginx/conf.d/default.conf
    rm -f nginx/conf.d/local.conf
    
    # Update domain in sanadtrade.conf
    sed -i "s/sanadtrade.com/$DOMAIN/g" nginx/conf.d/sanadtrade.conf
    
    # Restart nginx
    print_info "Restarting nginx..."
    docker-compose up -d nginx
    
    print_success "SSL setup complete!"
    echo ""
    echo "Your site is now available at:"
    echo "  https://$DOMAIN"
    echo "  https://www.$DOMAIN"
    echo ""
else
    print_error "Failed to obtain SSL certificate"
    print_warning "Make sure:"
    print_warning "  1. Domain DNS is pointing to this server"
    print_warning "  2. Port 80 is open and accessible"
    print_warning "  3. No other service is using port 80"
    exit 1
fi

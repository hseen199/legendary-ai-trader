#!/bin/bash
# ============================================
# SanadTrade - Update Script
# سكريبت التحديث
# ============================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_DIR

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              SanadTrade Update Script                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create backup first
print_info "Creating backup before update..."
./scripts/backup.sh

# Pull latest changes
print_info "Pulling latest changes from Git..."
git pull origin main

# Rebuild and restart containers
print_info "Rebuilding containers..."
docker-compose build --no-cache

print_info "Restarting services..."
docker-compose down
docker-compose up -d

# Run migrations
print_info "Running database migrations..."
sleep 10
docker-compose exec -T backend alembic upgrade head 2>/dev/null || true

# Health check
print_info "Checking services..."
sleep 5
docker-compose ps

print_success "Update completed!"
echo ""
print_info "Check logs with: docker-compose logs -f"

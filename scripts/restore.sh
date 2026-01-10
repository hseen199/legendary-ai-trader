#!/bin/bash
# ============================================
# SanadTrade - Restore Script
# سكريبت الاستعادة من النسخ الاحتياطي
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

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./restore.sh <backup_file.tar.gz>"
    echo ""
    echo "Available backups:"
    ls -lh $(dirname $(dirname $(realpath $0)))/backups/*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE=$1
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
TEMP_DIR="/tmp/sanadtrade_restore_$$"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    print_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

print_warning "This will restore from backup and OVERWRITE current data!"
read -p "Are you sure? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    print_info "Restore cancelled"
    exit 0
fi

# Create temp directory
mkdir -p $TEMP_DIR

# Extract backup
print_info "Extracting backup..."
tar -xzf "$BACKUP_FILE" -C $TEMP_DIR

# Find the backup name
BACKUP_NAME=$(ls $TEMP_DIR | grep "_db.sql" | sed 's/_db.sql//')

# Restore database
print_info "Restoring database..."
docker-compose -f $PROJECT_DIR/docker-compose.yml exec -T postgres psql -U sanadtrade -d sanadtrade < "$TEMP_DIR/${BACKUP_NAME}_db.sql"
print_success "Database restored"

# Restore configuration
if [ -d "$TEMP_DIR/$BACKUP_NAME" ]; then
    print_info "Restoring configuration..."
    cp "$TEMP_DIR/$BACKUP_NAME/.env" $PROJECT_DIR/ 2>/dev/null || true
    cp "$TEMP_DIR/$BACKUP_NAME/backend.env" $PROJECT_DIR/backend/.env 2>/dev/null || true
    print_success "Configuration restored"
fi

# Restore SSL certificates
if [ -f "$TEMP_DIR/${BACKUP_NAME}_ssl.tar.gz" ]; then
    print_info "Restoring SSL certificates..."
    tar -xzf "$TEMP_DIR/${BACKUP_NAME}_ssl.tar.gz" -C $PROJECT_DIR
    print_success "SSL certificates restored"
fi

# Cleanup
rm -rf $TEMP_DIR

print_success "Restore completed!"
print_warning "Please restart services: docker-compose restart"

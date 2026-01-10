#!/bin/bash
# ============================================
# SanadTrade - Backup Script
# سكريبت النسخ الاحتياطي
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

# Configuration
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
BACKUP_DIR="$PROJECT_DIR/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="sanadtrade_backup_$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

print_info "Starting backup..."

# Backup database
print_info "Backing up database..."
docker-compose -f $PROJECT_DIR/docker-compose.yml exec -T postgres pg_dump -U sanadtrade sanadtrade > "$BACKUP_DIR/${BACKUP_NAME}_db.sql"
print_success "Database backed up"

# Backup .env files
print_info "Backing up configuration..."
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
cp $PROJECT_DIR/.env "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
cp $PROJECT_DIR/backend/.env "$BACKUP_DIR/$BACKUP_NAME/backend.env" 2>/dev/null || true
print_success "Configuration backed up"

# Backup SSL certificates
if [ -d "$PROJECT_DIR/certbot/conf" ]; then
    print_info "Backing up SSL certificates..."
    tar -czf "$BACKUP_DIR/${BACKUP_NAME}_ssl.tar.gz" -C $PROJECT_DIR certbot/conf
    print_success "SSL certificates backed up"
fi

# Create final archive
print_info "Creating backup archive..."
cd $BACKUP_DIR
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}_db.sql" "$BACKUP_NAME" "${BACKUP_NAME}_ssl.tar.gz" 2>/dev/null || \
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}_db.sql" "$BACKUP_NAME"

# Cleanup temporary files
rm -f "${BACKUP_NAME}_db.sql"
rm -rf "$BACKUP_NAME"
rm -f "${BACKUP_NAME}_ssl.tar.gz"

# Keep only last 7 backups
ls -t $BACKUP_DIR/*.tar.gz 2>/dev/null | tail -n +8 | xargs -r rm

print_success "Backup completed: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"

# Show backup size
BACKUP_SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)
print_info "Backup size: $BACKUP_SIZE"

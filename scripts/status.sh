#!/bin/bash
# ============================================
# SanadTrade - Status Check Script
# سكريبت فحص حالة الخدمات
# ============================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_DIR

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              SanadTrade System Status                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Docker services status
echo -e "${BLUE}📦 Docker Services:${NC}"
echo "─────────────────────────────────────────────────────────────────"
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Check API health
echo -e "${BLUE}🔍 API Health Check:${NC}"
echo "─────────────────────────────────────────────────────────────────"
API_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo "FAILED")
if [[ "$API_RESPONSE" == *"healthy"* ]]; then
    echo -e "${GREEN}✅ Backend API: Healthy${NC}"
else
    echo -e "${RED}❌ Backend API: Not responding${NC}"
fi

FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null || echo "000")
if [ "$FRONTEND_RESPONSE" == "200" ]; then
    echo -e "${GREEN}✅ Frontend: Running${NC}"
else
    echo -e "${RED}❌ Frontend: Not responding (HTTP $FRONTEND_RESPONSE)${NC}"
fi
echo ""

# Database status
echo -e "${BLUE}🗄️ Database:${NC}"
echo "─────────────────────────────────────────────────────────────────"
DB_STATUS=$(docker-compose exec -T postgres pg_isready -U sanadtrade 2>/dev/null && echo "OK" || echo "FAILED")
if [[ "$DB_STATUS" == *"OK"* ]]; then
    echo -e "${GREEN}✅ PostgreSQL: Connected${NC}"
    
    # Get user count
    USER_COUNT=$(docker-compose exec -T postgres psql -U sanadtrade -t -c "SELECT COUNT(*) FROM users;" 2>/dev/null | tr -d ' ' || echo "N/A")
    echo "   Users: $USER_COUNT"
    
    # Get transaction count
    TX_COUNT=$(docker-compose exec -T postgres psql -U sanadtrade -t -c "SELECT COUNT(*) FROM transactions;" 2>/dev/null | tr -d ' ' || echo "N/A")
    echo "   Transactions: $TX_COUNT"
else
    echo -e "${RED}❌ PostgreSQL: Not connected${NC}"
fi
echo ""

# Redis status
echo -e "${BLUE}📮 Redis:${NC}"
echo "─────────────────────────────────────────────────────────────────"
REDIS_STATUS=$(docker-compose exec -T redis redis-cli ping 2>/dev/null || echo "FAILED")
if [[ "$REDIS_STATUS" == *"PONG"* ]]; then
    echo -e "${GREEN}✅ Redis: Connected${NC}"
else
    echo -e "${RED}❌ Redis: Not connected${NC}"
fi
echo ""

# Disk usage
echo -e "${BLUE}💾 Disk Usage:${NC}"
echo "─────────────────────────────────────────────────────────────────"
df -h / | tail -1 | awk '{print "   Used: " $3 " / " $2 " (" $5 " used)"}'
echo ""

# Memory usage
echo -e "${BLUE}🧠 Memory Usage:${NC}"
echo "─────────────────────────────────────────────────────────────────"
free -h | grep Mem | awk '{print "   Used: " $3 " / " $2}'
echo ""

# Docker disk usage
echo -e "${BLUE}🐳 Docker Disk Usage:${NC}"
echo "─────────────────────────────────────────────────────────────────"
docker system df 2>/dev/null | head -5
echo ""

# Recent logs
echo -e "${BLUE}📋 Recent Backend Logs:${NC}"
echo "─────────────────────────────────────────────────────────────────"
docker-compose logs --tail=5 backend 2>/dev/null | tail -5
echo ""

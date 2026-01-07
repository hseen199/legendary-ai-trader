#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#                    نظام التداول الخارق - سكريبت التشغيل
#                    Legendary Trading System - Run Script
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# الألوان
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# الشعار
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║     🚀 نظام التداول الخارق V2 - Legendary Trading System                      ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# التحقق من Python
echo -e "${YELLOW}[1/5] التحقق من Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 غير مثبت!${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# التحقق من البيئة الافتراضية
echo -e "${YELLOW}[2/5] التحقق من البيئة الافتراضية...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}إنشاء بيئة افتراضية...${NC}"
    python3 -m venv venv
fi

# تفعيل البيئة الافتراضية
source venv/bin/activate
echo -e "${GREEN}✅ البيئة الافتراضية مفعّلة${NC}"

# تثبيت المتطلبات
echo -e "${YELLOW}[3/5] التحقق من المتطلبات...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✅ المتطلبات مثبتة${NC}"

# التحقق من ملف .env
echo -e "${YELLOW}[4/5] التحقق من الإعدادات...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}⚠️ ملف .env غير موجود!${NC}"
    echo -e "${YELLOW}إنشاء ملف .env من المثال...${NC}"
    cp .env.example .env
    echo -e "${RED}❗ يرجى تعديل ملف .env وإضافة مفاتيح API${NC}"
    exit 1
fi
echo -e "${GREEN}✅ ملف الإعدادات موجود${NC}"

# إنشاء المجلدات المطلوبة
echo -e "${YELLOW}[5/5] إنشاء المجلدات...${NC}"
mkdir -p data logs models checkpoints
echo -e "${GREEN}✅ المجلدات جاهزة${NC}"

# التشغيل
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                         🚀 بدء تشغيل النظام...                                ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

python3 main.py

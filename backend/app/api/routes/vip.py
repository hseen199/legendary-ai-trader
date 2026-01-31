"""
VIP Routes - مسارات API لنظام VIP
يُضاف إلى /opt/asinax/backend/app/api/routes/vip.py
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/vip", tags=["VIP"])


# ============ Schemas ============

class VIPBenefitResponse(BaseModel):
    name_ar: str
    name_en: str
    value: str
    enabled: bool
    icon: str


class VIPLevelInfoResponse(BaseModel):
    key: str
    name_ar: str
    name_en: str
    icon: str
    color: str
    min_deposit: float
    max_deposit: Optional[float]
    performance_fee: float
    referral_bonus: float
    priority_support: bool
    weekly_reports: bool
    daily_reports: bool
    dedicated_manager: bool
    early_access: bool


class NextLevelResponse(BaseModel):
    level: str
    name_ar: str
    name_en: str
    min_deposit: float
    amount_needed: float
    progress: float
    icon: str
    color: str


class VIPInfoResponse(BaseModel):
    current_level: str
    level_name_ar: str
    level_name_en: str
    icon: str
    color: str
    performance_fee: float
    referral_bonus: float
    total_deposited: float
    next_level: Optional[NextLevelResponse]
    benefits: List[VIPBenefitResponse]


class VIPStatsResponse(BaseModel):
    total_users: int
    distribution: dict


class VIPUpgradeCheckResponse(BaseModel):
    old_level: str
    new_level: str
    upgraded: bool
    message: str


# ============ VIP Levels Data ============

VIP_LEVELS = {
    "bronze": {
        "name_ar": "برونزي",
        "name_en": "Bronze",
        "min_deposit": 0,
        "max_deposit": 999,
        "performance_fee": 20.0,
        "priority_support": False,
        "weekly_reports": False,
        "daily_reports": False,
        "dedicated_manager": False,
        "early_access": False,
        "referral_bonus": 5.0,
        "withdrawal_priority": 1,
        "color": "#CD7F32",
        "icon": "🥉"
    },
    "silver": {
        "name_ar": "فضي",
        "name_en": "Silver",
        "min_deposit": 1000,
        "max_deposit": 4999,
        "performance_fee": 18.0,
        "priority_support": True,
        "weekly_reports": True,
        "daily_reports": False,
        "dedicated_manager": False,
        "early_access": False,
        "referral_bonus": 7.0,
        "withdrawal_priority": 2,
        "color": "#C0C0C0",
        "icon": "🥈"
    },
    "gold": {
        "name_ar": "ذهبي",
        "name_en": "Gold",
        "min_deposit": 5000,
        "max_deposit": 24999,
        "performance_fee": 15.0,
        "priority_support": True,
        "weekly_reports": True,
        "daily_reports": True,
        "dedicated_manager": False,
        "early_access": True,
        "referral_bonus": 10.0,
        "withdrawal_priority": 3,
        "color": "#FFD700",
        "icon": "🥇"
    },
    "platinum": {
        "name_ar": "بلاتيني",
        "name_en": "Platinum",
        "min_deposit": 25000,
        "max_deposit": 99999,
        "performance_fee": 12.0,
        "priority_support": True,
        "weekly_reports": True,
        "daily_reports": True,
        "dedicated_manager": True,
        "early_access": True,
        "referral_bonus": 12.0,
        "withdrawal_priority": 4,
        "color": "#E5E4E2",
        "icon": "💎"
    },
    "diamond": {
        "name_ar": "ماسي",
        "name_en": "Diamond",
        "min_deposit": 100000,
        "max_deposit": None,
        "performance_fee": 10.0,
        "priority_support": True,
        "weekly_reports": True,
        "daily_reports": True,
        "dedicated_manager": True,
        "early_access": True,
        "referral_bonus": 15.0,
        "withdrawal_priority": 5,
        "color": "#B9F2FF",
        "icon": "💠"
    }
}


# ============ Helper Functions ============

def get_vip_level_by_deposit(total_deposited: float) -> str:
    """تحديد مستوى VIP بناءً على إجمالي الإيداعات"""
    for level_key in ["diamond", "platinum", "gold", "silver", "bronze"]:
        level = VIP_LEVELS[level_key]
        if total_deposited >= level["min_deposit"]:
            return level_key
    return "bronze"


def get_next_level_info(current_level: str, total_deposited: float) -> Optional[dict]:
    """الحصول على معلومات المستوى التالي"""
    levels_order = ["bronze", "silver", "gold", "platinum", "diamond"]
    
    try:
        current_index = levels_order.index(current_level)
    except ValueError:
        current_index = 0
    
    if current_index >= len(levels_order) - 1:
        return None
    
    next_level_key = levels_order[current_index + 1]
    next_level = VIP_LEVELS[next_level_key]
    
    amount_needed = next_level["min_deposit"] - total_deposited
    progress = (total_deposited / next_level["min_deposit"]) * 100 if next_level["min_deposit"] > 0 else 100
    
    return {
        "level": next_level_key,
        "name_ar": next_level["name_ar"],
        "name_en": next_level["name_en"],
        "min_deposit": next_level["min_deposit"],
        "amount_needed": max(0, amount_needed),
        "progress": min(100, progress),
        "icon": next_level["icon"],
        "color": next_level["color"]
    }


def get_user_benefits(level: str) -> List[dict]:
    """الحصول على قائمة مزايا المستخدم"""
    level_info = VIP_LEVELS.get(level, VIP_LEVELS["bronze"])
    
    benefits = [
        {
            "name_ar": "رسوم الأداء",
            "name_en": "Performance Fee",
            "value": f"{level_info['performance_fee']}%",
            "enabled": True,
            "icon": "💰"
        },
        {
            "name_ar": "دعم أولوي",
            "name_en": "Priority Support",
            "value": "متاح" if level_info["priority_support"] else "غير متاح",
            "enabled": level_info["priority_support"],
            "icon": "🎧"
        },
        {
            "name_ar": "تقارير أسبوعية",
            "name_en": "Weekly Reports",
            "value": "متاح" if level_info["weekly_reports"] else "غير متاح",
            "enabled": level_info["weekly_reports"],
            "icon": "📊"
        },
        {
            "name_ar": "تقارير يومية",
            "name_en": "Daily Reports",
            "value": "متاح" if level_info["daily_reports"] else "غير متاح",
            "enabled": level_info["daily_reports"],
            "icon": "📈"
        },
        {
            "name_ar": "مدير حساب مخصص",
            "name_en": "Dedicated Manager",
            "value": "متاح" if level_info["dedicated_manager"] else "غير متاح",
            "enabled": level_info["dedicated_manager"],
            "icon": "👤"
        },
        {
            "name_ar": "وصول مبكر للميزات",
            "name_en": "Early Access",
            "value": "متاح" if level_info["early_access"] else "غير متاح",
            "enabled": level_info["early_access"],
            "icon": "🚀"
        },
        {
            "name_ar": "مكافأة الإحالة",
            "name_en": "Referral Bonus",
            "value": f"{level_info['referral_bonus']}%",
            "enabled": True,
            "icon": "🎁"
        }
    ]
    
    return benefits


# ============ Endpoints ============

@router.get("/info", response_model=VIPInfoResponse)
async def get_vip_info(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    الحصول على معلومات VIP للمستخدم الحالي
    """
    level = current_user.vip_level or "bronze"
    level_info = VIP_LEVELS.get(level, VIP_LEVELS["bronze"])
    total_deposited = float(current_user.total_deposited or 0)
    
    benefits = get_user_benefits(level)
    next_level = get_next_level_info(level, total_deposited)
    
    return VIPInfoResponse(
        current_level=level,
        level_name_ar=level_info["name_ar"],
        level_name_en=level_info["name_en"],
        icon=level_info["icon"],
        color=level_info["color"],
        performance_fee=level_info["performance_fee"],
        referral_bonus=level_info["referral_bonus"],
        total_deposited=total_deposited,
        next_level=NextLevelResponse(**next_level) if next_level else None,
        benefits=[VIPBenefitResponse(**b) for b in benefits]
    )


@router.get("/levels", response_model=List[VIPLevelInfoResponse])
async def get_all_vip_levels():
    """
    الحصول على جميع مستويات VIP
    """
    return [
        VIPLevelInfoResponse(
            key=key,
            name_ar=level["name_ar"],
            name_en=level["name_en"],
            icon=level["icon"],
            color=level["color"],
            min_deposit=level["min_deposit"],
            max_deposit=level["max_deposit"],
            performance_fee=level["performance_fee"],
            referral_bonus=level["referral_bonus"],
            priority_support=level["priority_support"],
            weekly_reports=level["weekly_reports"],
            daily_reports=level["daily_reports"],
            dedicated_manager=level["dedicated_manager"],
            early_access=level["early_access"]
        )
        for key, level in VIP_LEVELS.items()
    ]


@router.post("/check-upgrade", response_model=VIPUpgradeCheckResponse)
async def check_and_upgrade_vip(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    التحقق وتحديث مستوى VIP للمستخدم
    """
    old_level = current_user.vip_level or "bronze"
    total_deposited = float(current_user.total_deposited or 0)
    
    new_level = get_vip_level_by_deposit(total_deposited)
    
    # تحديث المستوى إذا تغير
    if new_level != old_level:
        current_user.vip_level = new_level
        await db.commit()
        
        levels_order = ["bronze", "silver", "gold", "platinum", "diamond"]
        old_index = levels_order.index(old_level) if old_level in levels_order else 0
        new_index = levels_order.index(new_level) if new_level in levels_order else 0
        upgraded = new_index > old_index
        
        if upgraded:
            new_info = VIP_LEVELS[new_level]
            message = f"تهانينا! تمت ترقيتك إلى مستوى {new_info['name_ar']} {new_info['icon']}"
        else:
            message = f"تم تحديث مستواك إلى {VIP_LEVELS[new_level]['name_ar']}"
    else:
        upgraded = False
        message = "مستواك الحالي لم يتغير"
    
    return VIPUpgradeCheckResponse(
        old_level=old_level,
        new_level=new_level,
        upgraded=upgraded,
        message=message
    )


@router.get("/stats", response_model=VIPStatsResponse)
async def get_vip_statistics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    إحصائيات توزيع المستخدمين على مستويات VIP (للأدمن)
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    stats = {}
    total_users = 0
    
    for level_key in VIP_LEVELS.keys():
        result = await db.execute(
            select(func.count(User.id)).where(User.vip_level == level_key)
        )
        count = result.scalar() or 0
        stats[level_key] = count
        total_users += count
    
    distribution = {}
    for level_key, count in stats.items():
        distribution[level_key] = {
            "count": count,
            "percentage": (count / total_users * 100) if total_users > 0 else 0,
            "name_ar": VIP_LEVELS[level_key]["name_ar"],
            "icon": VIP_LEVELS[level_key]["icon"]
        }
    
    return VIPStatsResponse(
        total_users=total_users,
        distribution=distribution
    )


@router.get("/benefits/{level}")
async def get_level_benefits(level: str):
    """
    الحصول على مزايا مستوى معين
    """
    if level not in VIP_LEVELS:
        raise HTTPException(status_code=404, detail="VIP level not found")
    
    return {
        "level": level,
        "info": VIP_LEVELS[level],
        "benefits": get_user_benefits(level)
    }


@router.get("/compare")
async def compare_vip_levels():
    """
    مقارنة جميع مستويات VIP
    """
    comparison = []
    
    for key, level in VIP_LEVELS.items():
        comparison.append({
            "level": key,
            "name_ar": level["name_ar"],
            "name_en": level["name_en"],
            "icon": level["icon"],
            "min_deposit": level["min_deposit"],
            "performance_fee": level["performance_fee"],
            "referral_bonus": level["referral_bonus"],
            "features": {
                "priority_support": level["priority_support"],
                "weekly_reports": level["weekly_reports"],
                "daily_reports": level["daily_reports"],
                "dedicated_manager": level["dedicated_manager"],
                "early_access": level["early_access"]
            }
        })
    
    return comparison

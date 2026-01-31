"""
VIP System Service - خدمة نظام VIP والمستويات
يُضاف إلى /opt/asinax/backend/app/services/vip_service.py
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
import logging

logger = logging.getLogger(__name__)


# ============ تعريف مستويات VIP ============

VIP_LEVELS = {
    "bronze": {
        "name_ar": "برونزي",
        "name_en": "Bronze",
        "min_deposit": 0,
        "max_deposit": 999,
        "performance_fee": 20.0,  # نسبة رسوم الأداء
        "priority_support": False,
        "weekly_reports": False,
        "daily_reports": False,
        "dedicated_manager": False,
        "early_access": False,
        "referral_bonus": 5.0,  # نسبة مكافأة الإحالة
        "withdrawal_priority": 1,  # أولوية السحب (1 = عادي)
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
        "max_deposit": float('inf'),
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


class VIPService:
    """
    خدمة إدارة نظام VIP والمستويات
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ============ تحديد المستوى ============
    
    def get_vip_level_by_deposit(self, total_deposited: float) -> str:
        """
        تحديد مستوى VIP بناءً على إجمالي الإيداعات
        """
        for level_key in ["diamond", "platinum", "gold", "silver", "bronze"]:
            level = VIP_LEVELS[level_key]
            if total_deposited >= level["min_deposit"]:
                return level_key
        return "bronze"
    
    def get_vip_level_info(self, level: str) -> Dict[str, Any]:
        """
        الحصول على معلومات مستوى VIP
        """
        return VIP_LEVELS.get(level, VIP_LEVELS["bronze"])
    
    async def update_user_vip_level(self, user_id: int) -> Tuple[str, str]:
        """
        تحديث مستوى VIP للمستخدم بناءً على إيداعاته
        يُرجع (المستوى القديم، المستوى الجديد)
        """
        from app.models import User, Investor
        
        # الحصول على المستخدم
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None, None
        
        old_level = user.vip_level or "bronze"
        
        # حساب إجمالي الإيداعات
        total_deposited = float(user.total_deposited or 0)
        
        # تحديد المستوى الجديد
        new_level = self.get_vip_level_by_deposit(total_deposited)
        
        # تحديث المستوى إذا تغير
        if new_level != old_level:
            user.vip_level = new_level
            await self.db.commit()
            
            # إرسال إشعار بالترقية
            if self._is_upgrade(old_level, new_level):
                await self._notify_vip_upgrade(user_id, old_level, new_level)
        
        return old_level, new_level
    
    def _is_upgrade(self, old_level: str, new_level: str) -> bool:
        """التحقق إذا كان التغيير ترقية"""
        levels_order = ["bronze", "silver", "gold", "platinum", "diamond"]
        old_index = levels_order.index(old_level) if old_level in levels_order else 0
        new_index = levels_order.index(new_level) if new_level in levels_order else 0
        return new_index > old_index
    
    # ============ حساب الرسوم ============
    
    def calculate_performance_fee(self, level: str, profit: float) -> float:
        """
        حساب رسوم الأداء بناءً على مستوى VIP
        """
        level_info = self.get_vip_level_info(level)
        fee_percentage = level_info["performance_fee"]
        return profit * (fee_percentage / 100)
    
    def get_referral_bonus_rate(self, level: str) -> float:
        """
        الحصول على نسبة مكافأة الإحالة
        """
        level_info = self.get_vip_level_info(level)
        return level_info["referral_bonus"]
    
    # ============ المزايا ============
    
    def get_user_benefits(self, level: str, language: str = "ar") -> List[Dict[str, Any]]:
        """
        الحصول على قائمة مزايا المستخدم
        """
        level_info = self.get_vip_level_info(level)
        
        benefits = []
        
        # رسوم الأداء
        benefits.append({
            "name_ar": "رسوم الأداء",
            "name_en": "Performance Fee",
            "value": f"{level_info['performance_fee']}%",
            "enabled": True,
            "icon": "💰"
        })
        
        # الدعم الأولوي
        benefits.append({
            "name_ar": "دعم أولوي",
            "name_en": "Priority Support",
            "value": "متاح" if level_info["priority_support"] else "غير متاح",
            "enabled": level_info["priority_support"],
            "icon": "🎧"
        })
        
        # التقارير الأسبوعية
        benefits.append({
            "name_ar": "تقارير أسبوعية",
            "name_en": "Weekly Reports",
            "value": "متاح" if level_info["weekly_reports"] else "غير متاح",
            "enabled": level_info["weekly_reports"],
            "icon": "📊"
        })
        
        # التقارير اليومية
        benefits.append({
            "name_ar": "تقارير يومية",
            "name_en": "Daily Reports",
            "value": "متاح" if level_info["daily_reports"] else "غير متاح",
            "enabled": level_info["daily_reports"],
            "icon": "📈"
        })
        
        # مدير حساب مخصص
        benefits.append({
            "name_ar": "مدير حساب مخصص",
            "name_en": "Dedicated Manager",
            "value": "متاح" if level_info["dedicated_manager"] else "غير متاح",
            "enabled": level_info["dedicated_manager"],
            "icon": "👤"
        })
        
        # الوصول المبكر
        benefits.append({
            "name_ar": "وصول مبكر للميزات",
            "name_en": "Early Access",
            "value": "متاح" if level_info["early_access"] else "غير متاح",
            "enabled": level_info["early_access"],
            "icon": "🚀"
        })
        
        # مكافأة الإحالة
        benefits.append({
            "name_ar": "مكافأة الإحالة",
            "name_en": "Referral Bonus",
            "value": f"{level_info['referral_bonus']}%",
            "enabled": True,
            "icon": "🎁"
        })
        
        return benefits
    
    def get_next_level_info(self, current_level: str, total_deposited: float) -> Optional[Dict[str, Any]]:
        """
        الحصول على معلومات المستوى التالي والمبلغ المطلوب للترقية
        """
        levels_order = ["bronze", "silver", "gold", "platinum", "diamond"]
        
        try:
            current_index = levels_order.index(current_level)
        except ValueError:
            current_index = 0
        
        # إذا كان في أعلى مستوى
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
    
    # ============ إحصائيات VIP ============
    
    async def get_vip_statistics(self) -> Dict[str, Any]:
        """
        إحصائيات توزيع المستخدمين على مستويات VIP
        """
        from app.models import User
        
        stats = {}
        total_users = 0
        
        for level_key in VIP_LEVELS.keys():
            result = await self.db.execute(
                select(func.count(User.id)).where(User.vip_level == level_key)
            )
            count = result.scalar() or 0
            stats[level_key] = count
            total_users += count
        
        # حساب النسب
        distribution = {}
        for level_key, count in stats.items():
            distribution[level_key] = {
                "count": count,
                "percentage": (count / total_users * 100) if total_users > 0 else 0,
                "info": VIP_LEVELS[level_key]
            }
        
        return {
            "total_users": total_users,
            "distribution": distribution
        }
    
    # ============ الإشعارات ============
    
    async def _notify_vip_upgrade(self, user_id: int, old_level: str, new_level: str):
        """
        إرسال إشعار بترقية VIP
        """
        from app.models import Notification
        
        old_info = VIP_LEVELS.get(old_level, VIP_LEVELS["bronze"])
        new_info = VIP_LEVELS.get(new_level, VIP_LEVELS["bronze"])
        
        title = f"🎉 تهانينا! ترقية إلى {new_info['icon']} {new_info['name_ar']}"
        message = f"""
مبروك! تمت ترقيتك من مستوى {old_info['name_ar']} إلى {new_info['name_ar']}!

مزاياك الجديدة:
• رسوم أداء مخفضة: {new_info['performance_fee']}%
• مكافأة إحالة: {new_info['referral_bonus']}%
"""
        
        if new_info["priority_support"]:
            message += "• دعم أولوي ✓\n"
        if new_info["weekly_reports"]:
            message += "• تقارير أسبوعية ✓\n"
        if new_info["daily_reports"]:
            message += "• تقارير يومية ✓\n"
        if new_info["dedicated_manager"]:
            message += "• مدير حساب مخصص ✓\n"
        
        notification = Notification(
            user_id=user_id,
            type="system",
            title=title,
            message=message.strip(),
            data={
                "old_level": old_level,
                "new_level": new_level,
                "type": "vip_upgrade"
            },
            is_read=False,
            created_at=datetime.utcnow()
        )
        self.db.add(notification)
        await self.db.commit()
        
        # إرسال بريد إلكتروني
        await self._send_upgrade_email(user_id, old_level, new_level)
    
    async def _send_upgrade_email(self, user_id: int, old_level: str, new_level: str):
        """
        إرسال بريد إلكتروني بالترقية
        """
        try:
            from app.models import User
            from app.services.email_service import EmailService
            
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user and user.email:
                new_info = VIP_LEVELS.get(new_level, VIP_LEVELS["bronze"])
                
                email_service = EmailService()
                await email_service.send_email(
                    to_email=user.email,
                    subject=f"ASINAX - تهانينا! ترقية إلى {new_info['name_ar']}",
                    html_content=f"""
                    <div style="font-family: Arial, sans-serif; direction: rtl; text-align: right; background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); color: #fff; padding: 40px;">
                        <div style="text-align: center; margin-bottom: 30px;">
                            <span style="font-size: 60px;">{new_info['icon']}</span>
                            <h1 style="color: {new_info['color']}; margin: 20px 0;">مبروك!</h1>
                            <h2>تمت ترقيتك إلى مستوى {new_info['name_ar']}</h2>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin: 20px 0;">
                            <h3 style="color: #10b981;">مزاياك الجديدة:</h3>
                            <ul style="line-height: 2;">
                                <li>رسوم أداء مخفضة: {new_info['performance_fee']}%</li>
                                <li>مكافأة إحالة: {new_info['referral_bonus']}%</li>
                                {'<li>دعم أولوي ✓</li>' if new_info['priority_support'] else ''}
                                {'<li>تقارير أسبوعية ✓</li>' if new_info['weekly_reports'] else ''}
                                {'<li>تقارير يومية ✓</li>' if new_info['daily_reports'] else ''}
                                {'<li>مدير حساب مخصص ✓</li>' if new_info['dedicated_manager'] else ''}
                            </ul>
                        </div>
                        
                        <p style="text-align: center; color: #666; margin-top: 30px;">
                            شكراً لثقتك في ASINAX
                        </p>
                    </div>
                    """
                )
        except Exception as e:
            logger.error(f"Failed to send VIP upgrade email: {e}")


# ============ API Endpoints للإضافة إلى routes ============

"""
# يُضاف إلى /opt/asinax/backend/app/api/routes/investor.py أو ملف جديد

@router.get("/vip/info")
async def get_vip_info(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    '''الحصول على معلومات VIP للمستخدم'''
    vip_service = VIPService(db)
    
    level = current_user.vip_level or "bronze"
    level_info = vip_service.get_vip_level_info(level)
    benefits = vip_service.get_user_benefits(level)
    next_level = vip_service.get_next_level_info(level, float(current_user.total_deposited or 0))
    
    return {
        "current_level": level,
        "level_info": level_info,
        "benefits": benefits,
        "next_level": next_level
    }

@router.get("/vip/levels")
async def get_all_vip_levels():
    '''الحصول على جميع مستويات VIP'''
    return VIP_LEVELS
"""

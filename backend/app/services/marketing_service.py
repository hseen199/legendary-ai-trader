"""
Marketing Service - خدمة التسويق
نظام الإحالات، VIP، والكوبونات
"""
import secrets
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
import logging

logger = logging.getLogger(__name__)


class ReferralService:
    """خدمة نظام الإحالات"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def generate_referral_code(self, user_id: int) -> str:
        """إنشاء كود إحالة للمستخدم"""
        from app.models.advanced_models import ReferralCode
        
        # التحقق من وجود كود سابق
        result = await self.db.execute(
            select(ReferralCode)
            .where(ReferralCode.user_id == user_id)
            .where(ReferralCode.is_active == True)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            return existing.code
        
        # إنشاء كود جديد
        code = self._generate_unique_code()
        
        referral_code = ReferralCode(
            user_id=user_id,
            code=code,
            bonus_percent=5.0,  # 5% مكافأة افتراضية
            is_active=True
        )
        
        self.db.add(referral_code)
        await self.db.commit()
        
        logger.info(f"Generated referral code {code} for user {user_id}")
        return code
    
    def _generate_unique_code(self, length: int = 8) -> str:
        """إنشاء كود فريد"""
        chars = string.ascii_uppercase + string.digits
        return ''.join(secrets.choice(chars) for _ in range(length))
    
    async def apply_referral_code(
        self,
        new_user_id: int,
        code: str
    ) -> Optional[Dict]:
        """تطبيق كود إحالة عند التسجيل"""
        from app.models.advanced_models import ReferralCode, Referral
        
        # البحث عن الكود
        result = await self.db.execute(
            select(ReferralCode)
            .where(ReferralCode.code == code.upper())
            .where(ReferralCode.is_active == True)
        )
        referral_code = result.scalar_one_or_none()
        
        if not referral_code:
            return None
        
        # التحقق من الحد الأقصى للاستخدام
        if referral_code.max_uses and referral_code.times_used >= referral_code.max_uses:
            return None
        
        # التحقق من أن المستخدم لا يحيل نفسه
        if referral_code.user_id == new_user_id:
            return None
        
        # إنشاء سجل الإحالة
        referral = Referral(
            referrer_id=referral_code.user_id,
            referred_id=new_user_id,
            code_id=referral_code.id,
            bonus_amount=0,  # سيتم حسابها عند الإيداع
            bonus_paid=False
        )
        
        self.db.add(referral)
        
        # تحديث عداد الاستخدام
        referral_code.times_used += 1
        
        await self.db.commit()
        
        logger.info(f"Referral code {code} applied for user {new_user_id}")
        
        return {
            "referrer_id": referral_code.user_id,
            "bonus_percent": referral_code.bonus_percent
        }
    
    async def process_referral_bonus(
        self,
        user_id: int,
        deposit_amount: float
    ) -> Optional[float]:
        """معالجة مكافأة الإحالة عند الإيداع"""
        from app.models.advanced_models import Referral, ReferralCode
        
        # البحث عن الإحالة
        result = await self.db.execute(
            select(Referral)
            .where(Referral.referred_id == user_id)
            .where(Referral.bonus_paid == False)
        )
        referral = result.scalar_one_or_none()
        
        if not referral:
            return None
        
        # جلب الكود للحصول على نسبة المكافأة
        result = await self.db.execute(
            select(ReferralCode)
            .where(ReferralCode.id == referral.code_id)
        )
        code = result.scalar_one_or_none()
        
        if not code:
            return None
        
        # حساب المكافأة
        bonus = deposit_amount * (code.bonus_percent / 100)
        
        # تحديث الإحالة
        referral.bonus_amount = bonus
        referral.referred_deposit = deposit_amount
        referral.bonus_paid = True
        
        # تحديث إجمالي المكافآت للكود
        code.total_earned += bonus
        
        await self.db.commit()
        
        logger.info(f"Processed referral bonus ${bonus} for referrer {referral.referrer_id}")
        
        return bonus
    
    async def get_user_referrals(self, user_id: int) -> Dict:
        """الحصول على إحالات المستخدم"""
        from app.models.advanced_models import ReferralCode, Referral
        
        # جلب الكود
        result = await self.db.execute(
            select(ReferralCode)
            .where(ReferralCode.user_id == user_id)
            .where(ReferralCode.is_active == True)
        )
        code = result.scalar_one_or_none()
        
        if not code:
            return {
                "code": None,
                "total_referrals": 0,
                "total_earned": 0,
                "referrals": []
            }
        
        # جلب الإحالات
        result = await self.db.execute(
            select(Referral)
            .where(Referral.code_id == code.id)
            .order_by(desc(Referral.created_at))
        )
        referrals = result.scalars().all()
        
        return {
            "code": code.code,
            "bonus_percent": code.bonus_percent,
            "total_referrals": len(referrals),
            "total_earned": code.total_earned,
            "referrals": [
                {
                    "user_id": r.referred_id,
                    "deposit": r.referred_deposit,
                    "bonus": r.bonus_amount,
                    "paid": r.bonus_paid,
                    "date": r.created_at.isoformat()
                }
                for r in referrals
            ]
        }
    
    async def get_all_referral_stats(self) -> Dict:
        """الحصول على إحصائيات الإحالات للأدمن"""
        from app.models.advanced_models import ReferralCode, Referral
        
        # إجمالي الأكواد
        result = await self.db.execute(
            select(func.count(ReferralCode.id))
        )
        total_codes = result.scalar() or 0
        
        # إجمالي الإحالات
        result = await self.db.execute(
            select(func.count(Referral.id))
        )
        total_referrals = result.scalar() or 0
        
        # إجمالي المكافآت المدفوعة
        result = await self.db.execute(
            select(func.sum(Referral.bonus_amount))
            .where(Referral.bonus_paid == True)
        )
        total_paid = result.scalar() or 0
        
        # أفضل المحيلين
        result = await self.db.execute(
            select(
                ReferralCode.user_id,
                ReferralCode.code,
                ReferralCode.times_used,
                ReferralCode.total_earned
            )
            .order_by(desc(ReferralCode.total_earned))
            .limit(10)
        )
        top_referrers = [
            {
                "user_id": row.user_id,
                "code": row.code,
                "referrals": row.times_used,
                "earned": row.total_earned
            }
            for row in result.all()
        ]
        
        return {
            "total_codes": total_codes,
            "total_referrals": total_referrals,
            "total_paid": round(total_paid, 2),
            "top_referrers": top_referrers
        }


class VIPService:
    """خدمة نظام VIP"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def initialize_vip_tiers(self):
        """تهيئة مستويات VIP الافتراضية"""
        from app.models.advanced_models import VIPTier, VIPLevel
        
        tiers = [
            {
                "level": VIPLevel.BRONZE,
                "name_ar": "برونزي",
                "name_en": "Bronze",
                "min_deposit": 0,
                "fee_discount": 0,
                "withdrawal_limit_daily": 5000,
                "priority_support": False,
                "early_access": False,
                "custom_referral_bonus": 5
            },
            {
                "level": VIPLevel.SILVER,
                "name_ar": "فضي",
                "name_en": "Silver",
                "min_deposit": 1000,
                "fee_discount": 10,
                "withdrawal_limit_daily": 10000,
                "priority_support": False,
                "early_access": False,
                "custom_referral_bonus": 7
            },
            {
                "level": VIPLevel.GOLD,
                "name_ar": "ذهبي",
                "name_en": "Gold",
                "min_deposit": 5000,
                "fee_discount": 20,
                "withdrawal_limit_daily": 25000,
                "priority_support": True,
                "early_access": False,
                "custom_referral_bonus": 10
            },
            {
                "level": VIPLevel.PLATINUM,
                "name_ar": "بلاتيني",
                "name_en": "Platinum",
                "min_deposit": 25000,
                "fee_discount": 30,
                "withdrawal_limit_daily": 50000,
                "priority_support": True,
                "early_access": True,
                "custom_referral_bonus": 12
            },
            {
                "level": VIPLevel.DIAMOND,
                "name_ar": "ماسي",
                "name_en": "Diamond",
                "min_deposit": 100000,
                "fee_discount": 50,
                "withdrawal_limit_daily": 100000,
                "priority_support": True,
                "early_access": True,
                "custom_referral_bonus": 15
            }
        ]
        
        for tier_data in tiers:
            result = await self.db.execute(
                select(VIPTier)
                .where(VIPTier.level == tier_data["level"])
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                tier = VIPTier(**tier_data)
                self.db.add(tier)
        
        await self.db.commit()
        logger.info("VIP tiers initialized")
    
    async def get_user_vip_level(self, user_id: int) -> Dict:
        """الحصول على مستوى VIP للمستخدم"""
        from app.models.advanced_models import UserVIP, VIPTier
        
        result = await self.db.execute(
            select(UserVIP)
            .join(VIPTier)
            .where(UserVIP.user_id == user_id)
            .where(or_(
                UserVIP.expires_at == None,
                UserVIP.expires_at > datetime.utcnow()
            ))
        )
        user_vip = result.scalar_one_or_none()
        
        if not user_vip:
            # إرجاع المستوى الافتراضي (برونزي)
            result = await self.db.execute(
                select(VIPTier)
                .where(VIPTier.level == "bronze")
            )
            tier = result.scalar_one_or_none()
            
            return {
                "level": "bronze",
                "name_ar": "برونزي",
                "name_en": "Bronze",
                "fee_discount": 0,
                "withdrawal_limit": 5000,
                "priority_support": False,
                "early_access": False
            }
        
        tier = user_vip.tier
        return {
            "level": tier.level.value,
            "name_ar": tier.name_ar,
            "name_en": tier.name_en,
            "fee_discount": tier.fee_discount,
            "withdrawal_limit": tier.withdrawal_limit_daily,
            "priority_support": tier.priority_support,
            "early_access": tier.early_access,
            "expires_at": user_vip.expires_at.isoformat() if user_vip.expires_at else None
        }
    
    async def upgrade_user_vip(
        self,
        user_id: int,
        level: str,
        expires_at: Optional[datetime] = None,
        is_manual: bool = True
    ) -> bool:
        """ترقية مستوى VIP للمستخدم"""
        from app.models.advanced_models import UserVIP, VIPTier, VIPLevel
        
        # جلب المستوى
        result = await self.db.execute(
            select(VIPTier)
            .where(VIPTier.level == VIPLevel(level))
        )
        tier = result.scalar_one_or_none()
        
        if not tier:
            return False
        
        # حذف المستوى السابق
        result = await self.db.execute(
            select(UserVIP)
            .where(UserVIP.user_id == user_id)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            await self.db.delete(existing)
        
        # إنشاء المستوى الجديد
        user_vip = UserVIP(
            user_id=user_id,
            tier_id=tier.id,
            expires_at=expires_at,
            is_manual=is_manual
        )
        
        self.db.add(user_vip)
        await self.db.commit()
        
        logger.info(f"Upgraded user {user_id} to VIP level {level}")
        return True
    
    async def auto_upgrade_vip(self, user_id: int, total_deposit: float):
        """ترقية VIP تلقائية بناءً على الإيداع"""
        from app.models.advanced_models import VIPTier
        
        # جلب المستوى المناسب
        result = await self.db.execute(
            select(VIPTier)
            .where(VIPTier.min_deposit <= total_deposit)
            .order_by(desc(VIPTier.min_deposit))
            .limit(1)
        )
        tier = result.scalar_one_or_none()
        
        if tier:
            await self.upgrade_user_vip(
                user_id=user_id,
                level=tier.level.value,
                is_manual=False
            )
    
    async def get_all_vip_stats(self) -> Dict:
        """الحصول على إحصائيات VIP للأدمن"""
        from app.models.advanced_models import UserVIP, VIPTier
        
        stats = {}
        
        result = await self.db.execute(select(VIPTier))
        tiers = result.scalars().all()
        
        for tier in tiers:
            result = await self.db.execute(
                select(func.count(UserVIP.id))
                .where(UserVIP.tier_id == tier.id)
            )
            count = result.scalar() or 0
            stats[tier.level.value] = {
                "name_ar": tier.name_ar,
                "count": count
            }
        
        return stats


class CouponService:
    """خدمة الكوبونات"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_coupon(
        self,
        code: str,
        coupon_type: str,
        value: float,
        description: Optional[str] = None,
        min_deposit: float = 0,
        max_discount: Optional[float] = None,
        max_uses: Optional[int] = None,
        valid_until: Optional[datetime] = None,
        created_by: Optional[int] = None
    ) -> Dict:
        """إنشاء كوبون جديد"""
        from app.models.advanced_models import Coupon, CouponType
        
        coupon = Coupon(
            code=code.upper(),
            type=CouponType(coupon_type),
            value=value,
            description=description,
            min_deposit=min_deposit,
            max_discount=max_discount,
            max_uses=max_uses,
            valid_until=valid_until,
            created_by=created_by,
            is_active=True
        )
        
        self.db.add(coupon)
        await self.db.commit()
        
        logger.info(f"Created coupon {code}")
        
        return {
            "id": coupon.id,
            "code": coupon.code,
            "type": coupon.type.value,
            "value": coupon.value
        }
    
    async def validate_coupon(
        self,
        code: str,
        user_id: int,
        deposit_amount: float
    ) -> Optional[Dict]:
        """التحقق من صلاحية الكوبون"""
        from app.models.advanced_models import Coupon, CouponUsage
        
        # البحث عن الكوبون
        result = await self.db.execute(
            select(Coupon)
            .where(Coupon.code == code.upper())
            .where(Coupon.is_active == True)
        )
        coupon = result.scalar_one_or_none()
        
        if not coupon:
            return {"valid": False, "error": "كوبون غير موجود"}
        
        # التحقق من تاريخ الصلاحية
        if coupon.valid_until and coupon.valid_until < datetime.utcnow():
            return {"valid": False, "error": "الكوبون منتهي الصلاحية"}
        
        # التحقق من الحد الأقصى للاستخدام
        if coupon.max_uses and coupon.times_used >= coupon.max_uses:
            return {"valid": False, "error": "تم استخدام الكوبون الحد الأقصى"}
        
        # التحقق من الحد الأدنى للإيداع
        if deposit_amount < coupon.min_deposit:
            return {"valid": False, "error": f"الحد الأدنى للإيداع ${coupon.min_deposit}"}
        
        # التحقق من استخدام سابق
        result = await self.db.execute(
            select(CouponUsage)
            .where(CouponUsage.coupon_id == coupon.id)
            .where(CouponUsage.user_id == user_id)
        )
        existing_usage = result.scalar_one_or_none()
        
        if existing_usage:
            return {"valid": False, "error": "تم استخدام هذا الكوبون مسبقاً"}
        
        # حساب الخصم
        if coupon.type.value == "percentage":
            discount = deposit_amount * (coupon.value / 100)
        else:
            discount = coupon.value
        
        # تطبيق الحد الأقصى للخصم
        if coupon.max_discount and discount > coupon.max_discount:
            discount = coupon.max_discount
        
        return {
            "valid": True,
            "coupon_id": coupon.id,
            "discount": round(discount, 2),
            "type": coupon.type.value,
            "value": coupon.value
        }
    
    async def apply_coupon(
        self,
        coupon_id: int,
        user_id: int,
        discount_amount: float
    ) -> bool:
        """تطبيق الكوبون"""
        from app.models.advanced_models import Coupon, CouponUsage
        
        # تسجيل الاستخدام
        usage = CouponUsage(
            coupon_id=coupon_id,
            user_id=user_id,
            discount_amount=discount_amount
        )
        self.db.add(usage)
        
        # تحديث عداد الاستخدام
        result = await self.db.execute(
            select(Coupon)
            .where(Coupon.id == coupon_id)
        )
        coupon = result.scalar_one_or_none()
        
        if coupon:
            coupon.times_used += 1
        
        await self.db.commit()
        
        logger.info(f"Applied coupon {coupon_id} for user {user_id}")
        return True
    
    async def get_all_coupons(self) -> List[Dict]:
        """الحصول على جميع الكوبونات"""
        from app.models.advanced_models import Coupon
        
        result = await self.db.execute(
            select(Coupon)
            .order_by(desc(Coupon.created_at))
        )
        coupons = result.scalars().all()
        
        return [
            {
                "id": c.id,
                "code": c.code,
                "type": c.type.value,
                "value": c.value,
                "description": c.description,
                "min_deposit": c.min_deposit,
                "max_discount": c.max_discount,
                "max_uses": c.max_uses,
                "times_used": c.times_used,
                "valid_until": c.valid_until.isoformat() if c.valid_until else None,
                "is_active": c.is_active,
                "created_at": c.created_at.isoformat()
            }
            for c in coupons
        ]
    
    async def toggle_coupon(self, coupon_id: int, is_active: bool) -> bool:
        """تفعيل/تعطيل الكوبون"""
        from app.models.advanced_models import Coupon
        
        result = await self.db.execute(
            select(Coupon)
            .where(Coupon.id == coupon_id)
        )
        coupon = result.scalar_one_or_none()
        
        if coupon:
            coupon.is_active = is_active
            await self.db.commit()
            return True
        
        return False

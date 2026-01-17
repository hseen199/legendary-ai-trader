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

# ============ إعدادات الإحالات ============
REFERRAL_BONUS_AMOUNT = 10.0  # مكافأة ثابتة $10


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
            bonus_percent=0,  # لم نعد نستخدم النسبة المئوية
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
    
    async def get_or_create_referral_code(self, user_id: int) -> str:
        """الحصول على كود الإحالة أو إنشاء واحد جديد"""
        return await self.generate_referral_code(user_id)
    
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
        
        # التحقق من أن المستخدم لم يُحال من قبل
        result = await self.db.execute(
            select(Referral)
            .where(Referral.referred_id == new_user_id)
        )
        existing_referral = result.scalar_one_or_none()
        if existing_referral:
            return None
        
        # إنشاء سجل الإحالة
        referral = Referral(
            referrer_id=referral_code.user_id,
            referred_id=new_user_id,
            code_id=referral_code.id,
            bonus_amount=REFERRAL_BONUS_AMOUNT,  # مكافأة ثابتة $10
            bonus_paid=False
        )
        
        self.db.add(referral)
        
        # تحديث عداد الاستخدام
        referral_code.times_used += 1
        
        await self.db.commit()
        
        logger.info(f"Referral code {code} applied for user {new_user_id}")
        
        return {
            "referrer_id": referral_code.user_id,
            "bonus_amount": REFERRAL_BONUS_AMOUNT
        }
    
    async def process_referral_bonus(
        self,
        user_id: int,
        deposit_amount: float
    ) -> Optional[float]:
        """معالجة مكافأة الإحالة عند أول إيداع"""
        from app.models.advanced_models import Referral, ReferralCode
        from app.models.user import User
        
        # البحث عن الإحالة غير المدفوعة
        result = await self.db.execute(
            select(Referral)
            .where(Referral.referred_id == user_id)
            .where(Referral.bonus_paid == False)
        )
        referral = result.scalar_one_or_none()
        
        if not referral:
            return None
        
        # المكافأة الثابتة $10
        bonus = REFERRAL_BONUS_AMOUNT
        
        # تحديث الإحالة
        referral.bonus_amount = bonus
        referral.referred_deposit = deposit_amount
        referral.bonus_paid = True
        referral.paid_at = datetime.utcnow()
        
        # إضافة المكافأة لرصيد المُحيل
        result = await self.db.execute(
            select(User)
            .where(User.id == referral.referrer_id)
        )
        referrer = result.scalar_one_or_none()
        
        if referrer:
            referrer.balance += bonus
            logger.info(f"Added ${bonus} referral bonus to user {referrer.id}")
        
        # تحديث إجمالي المكافآت للكود
        result = await self.db.execute(
            select(ReferralCode)
            .where(ReferralCode.id == referral.code_id)
        )
        code = result.scalar_one_or_none()
        if code:
            code.total_earned += bonus
        
        await self.db.commit()
        
        logger.info(f"Processed referral bonus ${bonus} for referrer {referral.referrer_id}")
        
        return bonus
    
    async def get_user_referrals(self, user_id: int) -> Dict:
        """الحصول على إحالات المستخدم"""
        from app.models.advanced_models import Referral, ReferralCode
        from app.models.user import User
        
        # جلب كود الإحالة
        result = await self.db.execute(
            select(ReferralCode)
            .where(ReferralCode.user_id == user_id)
            .where(ReferralCode.is_active == True)
        )
        code = result.scalar_one_or_none()
        
        # جلب الإحالات
        result = await self.db.execute(
            select(Referral, User)
            .join(User, Referral.referred_id == User.id)
            .where(Referral.referrer_id == user_id)
            .order_by(desc(Referral.created_at))
        )
        referrals_data = result.all()
        
        referrals = []
        total_earned = 0
        pending_earnings = 0
        active_referrals = 0
        
        for referral, referred_user in referrals_data:
            referrals.append({
                "id": referral.id,
                "user_email": referred_user.email[:3] + "***" + referred_user.email[referred_user.email.index("@"):],
                "user_name": referred_user.full_name or "مستخدم",
                "status": "مكتمل" if referral.bonus_paid else "في انتظار الإيداع",
                "bonus": referral.bonus_amount,
                "deposit": referral.referred_deposit or 0,
                "created_at": referral.created_at.isoformat(),
                "paid_at": referral.paid_at.isoformat() if referral.paid_at else None
            })
            
            if referral.bonus_paid:
                total_earned += referral.bonus_amount
                active_referrals += 1
            else:
                pending_earnings += REFERRAL_BONUS_AMOUNT
        
        return {
            "referral_code": code.code if code else None,
            "referral_link": f"https://asinax.cloud/register?ref={code.code}" if code else None,
            "bonus_amount": REFERRAL_BONUS_AMOUNT,
            "total_referrals": len(referrals),
            "active_referrals": active_referrals,
            "total_earned": total_earned,
            "pending_earnings": pending_earnings,
            "referrals": referrals
        }
    
    async def get_referral_earnings(self, user_id: int) -> Dict:
        """أرباح الإحالة"""
        from app.models.advanced_models import Referral
        
        # إجمالي الأرباح المدفوعة
        result = await self.db.execute(
            select(func.sum(Referral.bonus_amount))
            .where(Referral.referrer_id == user_id)
            .where(Referral.bonus_paid == True)
        )
        total_earned = result.scalar() or 0
        
        # الأرباح المعلقة
        result = await self.db.execute(
            select(func.count(Referral.id))
            .where(Referral.referrer_id == user_id)
            .where(Referral.bonus_paid == False)
        )
        pending_count = result.scalar() or 0
        pending_earnings = pending_count * REFERRAL_BONUS_AMOUNT
        
        return {
            "total_earned": total_earned,
            "pending_earnings": pending_earnings,
            "bonus_per_referral": REFERRAL_BONUS_AMOUNT
        }
    
    async def apply_referral(self, user_id: int, code: str) -> Dict:
        """تطبيق كود إحالة"""
        result = await self.apply_referral_code(user_id, code)
        if result:
            return {"success": True, "message": f"تم تطبيق كود الإحالة. ستحصل على ${REFERRAL_BONUS_AMOUNT} بعد أول إيداع."}
        return {"error": "كود الإحالة غير صالح أو تم استخدامه مسبقاً"}
    
    async def get_referral_stats(self) -> Dict:
        """إحصائيات نظام الإحالة (للأدمن)"""
        from app.models.advanced_models import Referral, ReferralCode
        
        # إجمالي الإحالات
        result = await self.db.execute(select(func.count(Referral.id)))
        total_referrals = result.scalar() or 0
        
        # الإحالات المكتملة
        result = await self.db.execute(
            select(func.count(Referral.id))
            .where(Referral.bonus_paid == True)
        )
        completed_referrals = result.scalar() or 0
        
        # إجمالي المكافآت المدفوعة
        result = await self.db.execute(
            select(func.sum(Referral.bonus_amount))
            .where(Referral.bonus_paid == True)
        )
        total_paid = result.scalar() or 0
        
        # أكواد الإحالة النشطة
        result = await self.db.execute(
            select(func.count(ReferralCode.id))
            .where(ReferralCode.is_active == True)
        )
        active_codes = result.scalar() or 0
        
        return {
            "total_referrals": total_referrals,
            "completed_referrals": completed_referrals,
            "pending_referrals": total_referrals - completed_referrals,
            "total_paid": total_paid,
            "active_codes": active_codes,
            "bonus_per_referral": REFERRAL_BONUS_AMOUNT
        }
    
    async def get_top_referrers(self, limit: int = 10) -> List[Dict]:
        """أفضل المحيلين"""
        from app.models.advanced_models import Referral
        from app.models.user import User
        
        result = await self.db.execute(
            select(
                Referral.referrer_id,
                func.count(Referral.id).label('count'),
                func.sum(Referral.bonus_amount).label('total')
            )
            .where(Referral.bonus_paid == True)
            .group_by(Referral.referrer_id)
            .order_by(desc('count'))
            .limit(limit)
        )
        top_referrers = result.all()
        
        referrers_list = []
        for referrer_id, count, total in top_referrers:
            result = await self.db.execute(
                select(User)
                .where(User.id == referrer_id)
            )
            user = result.scalar_one_or_none()
            if user:
                referrers_list.append({
                    "user_id": referrer_id,
                    "email": user.email[:3] + "***" + user.email[user.email.index("@"):],
                    "name": user.full_name or "مستخدم",
                    "referrals_count": count,
                    "total_earned": total or 0
                })
        
        return referrers_list
    
    async def update_settings(self, commission_rate: float) -> Dict:
        """تحديث إعدادات الإحالة (للأدمن)"""
        # في النظام الجديد المكافأة ثابتة
        return {
            "success": True,
            "message": "نظام الإحالات يستخدم مكافأة ثابتة $10",
            "bonus_amount": REFERRAL_BONUS_AMOUNT
        }


class VIPService:
    """خدمة نظام VIP"""
    
    VIP_TIERS = {
        "bronze": {"min_referrals": 5, "bonus_multiplier": 1.0, "name": "برونزي"},
        "silver": {"min_referrals": 15, "bonus_multiplier": 1.2, "name": "فضي"},
        "gold": {"min_referrals": 30, "bonus_multiplier": 1.5, "name": "ذهبي"},
        "platinum": {"min_referrals": 50, "bonus_multiplier": 2.0, "name": "بلاتيني"}
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_vip_tiers(self) -> List[Dict]:
        """الحصول على مستويات VIP"""
        return [
            {
                "tier": tier,
                "name": info["name"],
                "min_referrals": info["min_referrals"],
                "bonus_multiplier": info["bonus_multiplier"],
                "bonus_amount": REFERRAL_BONUS_AMOUNT * info["bonus_multiplier"]
            }
            for tier, info in self.VIP_TIERS.items()
        ]
    
    async def get_user_vip_status(self, user_id: int) -> Dict:
        """حالة VIP للمستخدم"""
        from app.models.user import User
        from app.models.advanced_models import Referral
        
        result = await self.db.execute(
            select(User)
            .where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return {"error": "المستخدم غير موجود"}
        
        # عدد الإحالات المكتملة
        result = await self.db.execute(
            select(func.count(Referral.id))
            .where(Referral.referrer_id == user_id)
            .where(Referral.bonus_paid == True)
        )
        referrals_count = result.scalar() or 0
        
        # تحديد المستوى
        current_tier = None
        next_tier = None
        for tier, info in self.VIP_TIERS.items():
            if referrals_count >= info["min_referrals"]:
                current_tier = tier
            elif next_tier is None:
                next_tier = tier
                break
        
        return {
            "user_id": user_id,
            "vip_level": user.vip_level or current_tier,
            "referrals_count": referrals_count,
            "current_tier": current_tier,
            "next_tier": next_tier,
            "next_tier_requirement": self.VIP_TIERS.get(next_tier, {}).get("min_referrals", 0) if next_tier else None
        }
    
    async def get_vip_users(self, tier: Optional[str] = None) -> List[Dict]:
        """قائمة مستخدمي VIP"""
        from app.models.user import User
        
        query = select(User).where(User.vip_level != None)
        if tier:
            query = query.where(User.vip_level == tier)
        
        result = await self.db.execute(query)
        users = result.scalars().all()
        
        return [
            {
                "id": u.id,
                "email": u.email,
                "name": u.full_name,
                "vip_level": u.vip_level
            }
            for u in users
        ]
    
    async def upgrade_user(self, user_id: int, tier: str, admin_id: int) -> bool:
        """ترقية مستخدم لـ VIP"""
        from app.models.user import User
        
        if tier not in self.VIP_TIERS:
            return False
        
        result = await self.db.execute(
            select(User)
            .where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.vip_level = tier
            await self.db.commit()
            logger.info(f"User {user_id} upgraded to VIP {tier} by admin {admin_id}")
            return True
        
        return False
    
    async def downgrade_user(self, user_id: int) -> bool:
        """تخفيض مستوى VIP"""
        from app.models.user import User
        
        result = await self.db.execute(
            select(User)
            .where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.vip_level = None
            await self.db.commit()
            return True
        
        return False
    
    async def get_vip_stats(self) -> Dict:
        """إحصائيات VIP"""
        from app.models.user import User
        
        stats = {}
        for tier in self.VIP_TIERS.keys():
            result = await self.db.execute(
                select(func.count(User.id))
                .where(User.vip_level == tier)
            )
            stats[tier] = result.scalar() or 0
        
        return {
            "tiers": stats,
            "total_vip_users": sum(stats.values())
        }


class CouponService:
    """خدمة الكوبونات"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_coupon(
        self,
        code: str,
        discount_type: str,
        discount_value: float,
        max_uses: Optional[int] = None,
        min_deposit: Optional[float] = None,
        expires_days: Optional[int] = None,
        created_by: int = None
    ) -> Dict:
        """إنشاء كوبون جديد"""
        from app.models.advanced_models import Coupon, CouponType
        
        # التحقق من عدم وجود كوبون بنفس الكود
        result = await self.db.execute(
            select(Coupon)
            .where(Coupon.code == code.upper())
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            return {"error": "كوبون بهذا الكود موجود مسبقاً"}
        
        valid_until = None
        if expires_days:
            valid_until = datetime.utcnow() + timedelta(days=expires_days)
        
        coupon = Coupon(
            code=code.upper(),
            type=CouponType.PERCENTAGE if discount_type == "percentage" else CouponType.FIXED,
            value=discount_value,
            min_deposit=min_deposit or 0,
            max_discount=None,
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
    ) -> Dict:
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
        code: str,
        user_id: int,
        amount: float
    ) -> Dict:
        """تطبيق الكوبون"""
        # التحقق أولاً
        validation = await self.validate_coupon(code, user_id, amount)
        if not validation.get("valid"):
            return validation
        
        from app.models.advanced_models import Coupon, CouponUsage
        
        coupon_id = validation["coupon_id"]
        discount = validation["discount"]
        
        # تسجيل الاستخدام
        usage = CouponUsage(
            coupon_id=coupon_id,
            user_id=user_id,
            discount_amount=discount
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
        
        logger.info(f"Applied coupon {code} for user {user_id}, discount: ${discount}")
        
        return {
            "success": True,
            "discount": discount,
            "final_amount": amount - discount
        }
    
    async def get_all_coupons(self, active_only: bool = False) -> List[Dict]:
        """الحصول على جميع الكوبونات"""
        from app.models.advanced_models import Coupon
        
        query = select(Coupon).order_by(desc(Coupon.created_at))
        if active_only:
            query = query.where(Coupon.is_active == True)
        
        result = await self.db.execute(query)
        coupons = result.scalars().all()
        
        return [
            {
                "id": c.id,
                "code": c.code,
                "type": c.type.value,
                "value": c.value,
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
    
    async def delete_coupon(self, coupon_id: int) -> bool:
        """حذف كوبون"""
        from app.models.advanced_models import Coupon
        
        result = await self.db.execute(
            select(Coupon)
            .where(Coupon.id == coupon_id)
        )
        coupon = result.scalar_one_or_none()
        
        if coupon:
            await self.db.delete(coupon)
            await self.db.commit()
            return True
        
        return False
    
    async def get_coupon_stats(self) -> Dict:
        """إحصائيات الكوبونات"""
        from app.models.advanced_models import Coupon, CouponUsage
        
        # إجمالي الكوبونات
        result = await self.db.execute(select(func.count(Coupon.id)))
        total_coupons = result.scalar() or 0
        
        # الكوبونات النشطة
        result = await self.db.execute(
            select(func.count(Coupon.id))
            .where(Coupon.is_active == True)
        )
        active_coupons = result.scalar() or 0
        
        # إجمالي الاستخدامات
        result = await self.db.execute(select(func.count(CouponUsage.id)))
        total_uses = result.scalar() or 0
        
        # إجمالي الخصومات
        result = await self.db.execute(
            select(func.sum(CouponUsage.discount_amount))
        )
        total_discounts = result.scalar() or 0
        
        return {
            "total_coupons": total_coupons,
            "active_coupons": active_coupons,
            "total_uses": total_uses,
            "total_discounts": total_discounts
        }

"""
Marketing Routes - مسارات التسويق والإحالات
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from pydantic import BaseModel

from app.core.database import get_db
from app.core.auth import get_current_admin, get_current_user
from app.services.marketing_service import ReferralService, VIPService, CouponService

router = APIRouter(prefix="/marketing", tags=["Marketing"])


# ============ Schemas ============

class CreateCouponRequest(BaseModel):
    code: str
    discount_type: str  # percentage, fixed
    discount_value: float
    max_uses: Optional[int] = None
    min_deposit: Optional[float] = None
    expires_days: Optional[int] = None


class UpdateVIPRequest(BaseModel):
    user_id: int
    tier: str  # bronze, silver, gold, platinum


# ============ Referral Endpoints ============

@router.get("/referral/stats")
async def get_referral_stats(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إحصائيات نظام الإحالة (للأدمن)"""
    service = ReferralService(db)
    return await service.get_referral_stats()


@router.get("/referral/top")
async def get_top_referrers(
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """أفضل المحيلين"""
    service = ReferralService(db)
    return await service.get_top_referrers(limit)


@router.get("/referral/my")
async def get_my_referrals(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """إحالاتي (للمستخدم)"""
    service = ReferralService(db)
    return await service.get_user_referrals(user.id)


@router.get("/referral/code")
async def get_my_referral_code(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """الحصول على كود الإحالة الخاص بي"""
    service = ReferralService(db)
    code = await service.get_or_create_referral_code(user.id)
    return {"referral_code": code}


@router.post("/referral/apply")
async def apply_referral_code(
    code: str,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """تطبيق كود إحالة"""
    service = ReferralService(db)
    result = await service.apply_referral(user.id, code)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/referral/earnings")
async def get_referral_earnings(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """أرباح الإحالة"""
    service = ReferralService(db)
    return await service.get_referral_earnings(user.id)


@router.post("/referral/settings")
async def update_referral_settings(
    commission_rate: float = Query(ge=0, le=0.1),
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """تحديث إعدادات الإحالة (للأدمن)"""
    service = ReferralService(db)
    return await service.update_settings(commission_rate)


# ============ VIP Endpoints ============

@router.get("/vip/tiers")
async def get_vip_tiers(
    db: AsyncSession = Depends(get_db)
):
    """الحصول على مستويات VIP"""
    service = VIPService(db)
    return await service.get_vip_tiers()


@router.get("/vip/my-status")
async def get_my_vip_status(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """حالة VIP الخاصة بي"""
    service = VIPService(db)
    return await service.get_user_vip_status(user.id)


@router.get("/vip/users")
async def get_vip_users(
    tier: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """قائمة مستخدمي VIP (للأدمن)"""
    service = VIPService(db)
    return await service.get_vip_users(tier)


@router.post("/vip/upgrade")
async def upgrade_user_vip(
    request: UpdateVIPRequest,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """ترقية مستخدم لـ VIP (للأدمن)"""
    service = VIPService(db)
    result = await service.upgrade_user(request.user_id, request.tier, admin.id)
    
    if not result:
        raise HTTPException(status_code=400, detail="فشل في ترقية المستخدم")
    
    return {"success": True, "message": f"تم ترقية المستخدم إلى {request.tier}"}


@router.post("/vip/downgrade")
async def downgrade_user_vip(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """تخفيض مستوى VIP (للأدمن)"""
    service = VIPService(db)
    result = await service.downgrade_user(user_id)
    
    if not result:
        raise HTTPException(status_code=400, detail="فشل في تخفيض المستخدم")
    
    return {"success": True}


@router.get("/vip/stats")
async def get_vip_stats(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إحصائيات VIP (للأدمن)"""
    service = VIPService(db)
    return await service.get_vip_stats()


# ============ Coupon Endpoints ============

@router.post("/coupons/create")
async def create_coupon(
    request: CreateCouponRequest,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إنشاء كوبون جديد"""
    service = CouponService(db)
    result = await service.create_coupon(
        code=request.code,
        discount_type=request.discount_type,
        discount_value=request.discount_value,
        max_uses=request.max_uses,
        min_deposit=request.min_deposit,
        expires_days=request.expires_days,
        created_by=admin.id
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/coupons")
async def get_all_coupons(
    active_only: bool = False,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """الحصول على جميع الكوبونات"""
    service = CouponService(db)
    return await service.get_all_coupons(active_only)


@router.post("/coupons/validate")
async def validate_coupon(
    code: str,
    amount: float,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """التحقق من صلاحية الكوبون"""
    service = CouponService(db)
    result = await service.validate_coupon(code, user.id, amount)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/coupons/apply")
async def apply_coupon(
    code: str,
    amount: float,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    """تطبيق الكوبون"""
    service = CouponService(db)
    result = await service.apply_coupon(code, user.id, amount)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.delete("/coupons/{coupon_id}")
async def delete_coupon(
    coupon_id: int,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """حذف كوبون"""
    service = CouponService(db)
    result = await service.delete_coupon(coupon_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="الكوبون غير موجود")
    
    return {"success": True}


@router.put("/coupons/{coupon_id}/toggle")
async def toggle_coupon(
    coupon_id: int,
    is_active: bool,
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """تفعيل/تعطيل كوبون"""
    service = CouponService(db)
    result = await service.toggle_coupon(coupon_id, is_active)
    
    if not result:
        raise HTTPException(status_code=404, detail="الكوبون غير موجود")
    
    return {"success": True}


@router.get("/coupons/stats")
async def get_coupon_stats(
    db: AsyncSession = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """إحصائيات الكوبونات"""
    service = CouponService(db)
    return await service.get_coupon_stats()

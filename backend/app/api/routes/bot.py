"""
مسارات API للتحكم بالبوت التداول
Bot Control API Routes
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List
import logging

from app.services.bot_integration_service import bot_service, BotStatus
from app.core.security import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bot", tags=["bot"])


@router.post("/start")
async def start_bot(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    بدء البوت التداول.
    
    يتطلب صلاحيات الأدمن.
    """
    # التحقق من صلاحيات الأدمن
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="لا توجد صلاحيات كافية"
        )
    
    try:
        success = await bot_service.start_bot()
        
        if success:
            return {
                "success": True,
                "message": "تم بدء البوت بنجاح",
                "status": (await bot_service.get_bot_status())
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="فشل في بدء البوت"
            )
    except Exception as e:
        logger.error(f"خطأ في بدء البوت: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ في بدء البوت: {str(e)}"
        )


@router.post("/stop")
async def stop_bot(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    إيقاف البوت التداول.
    
    يتطلب صلاحيات الأدمن.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="لا توجد صلاحيات كافية"
        )
    
    try:
        success = await bot_service.stop_bot()
        
        if success:
            return {
                "success": True,
                "message": "تم إيقاف البوت بنجاح",
                "status": (await bot_service.get_bot_status())
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="فشل في إيقاف البوت"
            )
    except Exception as e:
        logger.error(f"خطأ في إيقاف البوت: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ في إيقاف البوت: {str(e)}"
        )


@router.post("/pause")
async def pause_bot(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    إيقاف البوت مؤقتاً (بدون إيقاف العملية).
    
    يتطلب صلاحيات الأدمن.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="لا توجد صلاحيات كافية"
        )
    
    try:
        success = await bot_service.pause_bot()
        
        if success:
            return {
                "success": True,
                "message": "تم إيقاف البوت مؤقتاً",
                "status": (await bot_service.get_bot_status())
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="فشل في إيقاف البوت مؤقتاً"
            )
    except Exception as e:
        logger.error(f"خطأ في إيقاف البوت مؤقتاً: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ: {str(e)}"
        )


@router.post("/resume")
async def resume_bot(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    استئناف البوت المتوقف مؤقتاً.
    
    يتطلب صلاحيات الأدمن.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="لا توجد صلاحيات كافية"
        )
    
    try:
        success = await bot_service.resume_bot()
        
        if success:
            return {
                "success": True,
                "message": "تم استئناف البوت",
                "status": (await bot_service.get_bot_status())
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="فشل في استئناف البوت"
            )
    except Exception as e:
        logger.error(f"خطأ في استئناف البوت: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ: {str(e)}"
        )


@router.get("/status")
async def get_bot_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    الحصول على حالة البوت الحالية.
    """
    try:
        status_data = await bot_service.get_bot_status()
        return {
            "success": True,
            "data": status_data
        }
    except Exception as e:
        logger.error(f"خطأ في الحصول على حالة البوت: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ: {str(e)}"
        )


@router.get("/performance")
async def get_bot_performance(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    الحصول على بيانات أداء البوت.
    """
    try:
        performance = await bot_service.get_bot_performance()
        return {
            "success": True,
            "data": performance
        }
    except Exception as e:
        logger.error(f"خطأ في الحصول على أداء البوت: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ: {str(e)}"
        )


@router.get("/trades")
async def get_bot_trades(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    الحصول على قائمة الصفقات الأخيرة.
    
    Parameters:
        limit: عدد الصفقات المطلوبة (الحد الأقصى 1000)
    """
    if limit > 1000:
        limit = 1000
    
    try:
        trades = await bot_service.get_bot_trades(limit=limit)
        return {
            "success": True,
            "count": len(trades),
            "data": trades
        }
    except Exception as e:
        logger.error(f"خطأ في الحصول على الصفقات: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ: {str(e)}"
        )


@router.get("/portfolio")
async def get_bot_portfolio(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    الحصول على تفاصيل محفظة البوت.
    """
    try:
        portfolio = await bot_service.get_bot_portfolio()
        return {
            "success": True,
            "data": portfolio
        }
    except Exception as e:
        logger.error(f"خطأ في الحصول على محفظة البوت: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطأ: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    فحص صحة البوت (بدون الحاجة للمصادقة).
    """
    try:
        is_healthy = await bot_service.health_check()
        status_data = await bot_service.get_bot_status()
        
        return {
            "success": True,
            "is_healthy": is_healthy,
            "status": status_data
        }
    except Exception as e:
        logger.error(f"خطأ في فحص صحة البوت: {e}")
        return {
            "success": False,
            "is_healthy": False,
            "error": str(e)
        }

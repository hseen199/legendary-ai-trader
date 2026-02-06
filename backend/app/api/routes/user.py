from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User
from app.models.investor import Investor
from pydantic import BaseModel, Field
from typing import Optional
import phonenumbers

router = APIRouter(prefix="/user", tags=["User"])

class AccountProgressResponse(BaseModel):
    email_verified: bool
    phone_verified: bool
    two_factor_enabled: bool
    first_deposit_made: bool
    progress_percent: int

@router.get("/progress", response_model=AccountProgressResponse)
async def get_account_progress(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate and return the user's account completion progress.
    """
    # 1. Email Verification
    email_verified = current_user.is_verified
    
    # 2. Phone Verification
    phone_verified = current_user.phone_number is not None and current_user.phone_verified
    
    # 3. Two-Factor Authentication
    two_factor_enabled = current_user.two_factor_enabled
    
    # 4. First Deposit - Check if user has investor record (async query to avoid lazy loading issue)
    investor_result = await db.execute(
        select(Investor).where(Investor.user_id == current_user.id)
    )
    investor = investor_result.scalar_one_or_none()
    first_deposit_made = investor is not None
    
    # Calculate progress percentage
    completed_steps = sum([email_verified, phone_verified, two_factor_enabled, first_deposit_made])
    total_steps = 4
    progress_percent = int((completed_steps / total_steps) * 100)
    
    return AccountProgressResponse(
        email_verified=email_verified,
        phone_verified=phone_verified,
        two_factor_enabled=two_factor_enabled,
        first_deposit_made=first_deposit_made,
        progress_percent=progress_percent,
    )

class ProfileUpdateRequest(BaseModel):
    phone_number: Optional[str] = Field(None, example="+14155552671")
    full_name: Optional[str] = Field(None, example="John Doe")

@router.put("/profile")
async def update_profile(
    profile_data: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user profile (phone number and/or full name).
    """
    try:
        updated_fields = []
        
        # Update phone number if provided
        if profile_data.phone_number:
            try:
                phone_number_parsed = phonenumbers.parse(profile_data.phone_number, None)
                if not phonenumbers.is_valid_number(phone_number_parsed):
                    raise HTTPException(status_code=400, detail="Invalid phone number format.")
                
                # Format to E.164 standard
                formatted_phone = phonenumbers.format_number(phone_number_parsed, phonenumbers.PhoneNumberFormat.E164)
                
                current_user.phone_number = formatted_phone
                current_user.phone_verified = True  # Auto-verify for now
                updated_fields.append("phone_number")
            except phonenumbers.phonenumberutil.NumberParseException:
                # If phone number is invalid, skip it
                pass
        
        # Update full name if provided
        if profile_data.full_name:
            current_user.full_name = profile_data.full_name
            updated_fields.append("full_name")
        
        if not updated_fields:
            raise HTTPException(status_code=400, detail="No fields to update.")
        
        db.add(current_user)
        await db.commit()
        await db.refresh(current_user)
        
        return {
            "message": "Profile updated successfully",
            "updated_fields": updated_fields,
            "phone_number": current_user.phone_number,
            "full_name": current_user.full_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile")
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user profile.
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "phone_number": current_user.phone_number,
        "phone_verified": current_user.phone_verified,
        "avatar_url": current_user.avatar_url,
        "is_verified": current_user.is_verified,
        "two_factor_enabled": current_user.two_factor_enabled,
        "vip_level": current_user.vip_level,
        "referral_code": current_user.referral_code,
        "created_at": current_user.created_at
    }

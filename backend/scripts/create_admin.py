#!/usr/bin/env python3
"""
Script to create the first admin user
Usage: python scripts/create_admin.py
"""

import asyncio
import sys
import os
import secrets
import string

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

from app.core.config import settings
from app.core.security import get_password_hash
from app.models.user import User, Balance


def generate_referral_code(length=8):
    """Generate a random referral code"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


async def create_admin_user(
    email: str,
    password: str,
    full_name: str = "Admin"
):
    """Create admin user in database"""
    
    # Create engine
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Check if user already exists
        result = await session.execute(
            select(User).where(User.email == email)
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            print(f"❌ User with email {email} already exists!")
            if not existing_user.is_admin:
                existing_user.is_admin = True
                await session.commit()
                print(f"✅ User {email} has been promoted to admin!")
            return
        
        # Create new admin user
        admin_user = User(
            email=email,
            password_hash=get_password_hash(password),
            full_name=full_name,
            status="active",
            is_admin=True,
            is_verified=True,
            referral_code=generate_referral_code(),
            vip_level="platinum"
        )
        session.add(admin_user)
        await session.flush()
        
        # Create balance record
        balance = Balance(
            user_id=admin_user.id,
            units=0.0,
            balance_usd=0.0
        )
        session.add(balance)
        
        await session.commit()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Admin User Created!                        ║
╠══════════════════════════════════════════════════════════════╣
║  Email:    {email:<48} ║
║  Password: {password:<48} ║
║  Name:     {full_name:<48} ║
║  Referral: {admin_user.referral_code:<48} ║
╚══════════════════════════════════════════════════════════════╝

⚠️  Please change the password after first login!
        """)
    
    await engine.dispose()


async def main():
    # Default admin credentials (should be changed after first login)
    admin_email = os.getenv("ADMIN_EMAIL", "admin@asinax.com")
    admin_password = os.getenv("ADMIN_PASSWORD", "Admin@123456")
    admin_name = os.getenv("ADMIN_NAME", "System Admin")
    
    print("\n🔐 Creating Admin User for Asinax...\n")
    
    await create_admin_user(
        email=admin_email,
        password=admin_password,
        full_name=admin_name
    )


if __name__ == "__main__":
    asyncio.run(main())

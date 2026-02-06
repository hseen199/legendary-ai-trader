import random
import string
from fastapi import Request

def generate_otp(length: int = 6) -> str:
    """Generate a random OTP code"""
    return ''.join(random.choices(string.digits, k=length))

def get_client_info(request: Request) -> dict:
    """Get client information from request"""
    return {
        "ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "origin": request.headers.get("origin", "unknown")
    }

"""
Security and Authentication for Aegis API
"""

import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Redis for caching
redis_client = None

async def get_redis():
    global redis_client
    if not redis_client:
        redis_client = await redis.from_url(settings.REDIS_URL)
    return redis_client

class APIKeyManager:
    """Manage API keys and authentication"""

    @staticmethod
    def generate_api_key() -> tuple[str, str]:
        """Generate a new API key pair (public_key, secret_key)"""
        public_key = f"pk_{secrets.token_urlsafe(20)}"
        secret_key = f"sk_{secrets.token_urlsafe(32)}"
        return public_key, secret_key

    @staticmethod
    def hash_secret_key(secret_key: str) -> str:
        """Hash a secret key for storage"""
        return pwd_context.hash(secret_key)

    @staticmethod
    def verify_secret_key(plain_key: str, hashed_key: str) -> bool:
        """Verify a secret key against its hash"""
        return pwd_context.verify(plain_key, hashed_key)

    @staticmethod
    def create_key_signature(api_key: str, timestamp: int) -> str:
        """Create HMAC signature for API key validation"""
        message = f"{api_key}{timestamp}".encode()
        signature = hmac.new(
            settings.SECRET_KEY.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature

async def verify_api_key(api_key: str) -> Optional[Dict]:
    """
    Verify API key and return customer data

    Args:
        api_key: The API key to verify

    Returns:
        Customer data if valid, None otherwise
    """
    try:
        # Check cache first
        cache = await get_redis()
        cached_data = await cache.get(f"api_key:{api_key}")

        if cached_data:
            import json
            return json.loads(cached_data)

        # Validate format
        if not (api_key.startswith("sk_") or api_key.startswith("pk_")):
            return None

        # In production, query database
        # For now, return mock data for valid format
        if api_key.startswith("sk_"):
            customer_data = {
                "customer_id": "cus_" + secrets.token_urlsafe(12),
                "api_key": api_key,
                "plan": "growth",
                "requests_limit": 500000000,  # 500M requests
                "requests_used": 150000,
                "created_at": datetime.utcnow().isoformat(),
                "is_active": True
            }

            # Cache for 5 minutes
            await cache.setex(
                f"api_key:{api_key}",
                300,
                json.dumps(customer_data, default=str)
            )

            return customer_data

        return None

    except Exception as e:
        print(f"API key verification error: {e}")
        return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict]:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        return None

class PermissionChecker:
    """Check API permissions based on plan"""

    PLAN_LIMITS = {
        "starter": {
            "requests_per_month": 10_000_000,
            "rate_limit_per_second": 100,
            "max_batch_size": 10,
            "features": ["redaction", "masking"]
        },
        "growth": {
            "requests_per_month": 500_000_000,
            "rate_limit_per_second": 1000,
            "max_batch_size": 100,
            "features": ["redaction", "masking", "tokenization", "differential_privacy"]
        },
        "enterprise": {
            "requests_per_month": 10_000_000_000,
            "rate_limit_per_second": 10000,
            "max_batch_size": 1000,
            "features": ["all"]
        }
    }

    @classmethod
    def check_feature_access(cls, plan: str, feature: str) -> bool:
        """Check if a plan has access to a feature"""
        plan_config = cls.PLAN_LIMITS.get(plan, cls.PLAN_LIMITS["starter"])

        if "all" in plan_config["features"]:
            return True

        return feature in plan_config["features"]

    @classmethod
    def get_rate_limit(cls, plan: str) -> int:
        """Get rate limit for a plan"""
        plan_config = cls.PLAN_LIMITS.get(plan, cls.PLAN_LIMITS["starter"])
        return plan_config["rate_limit_per_second"]

    @classmethod
    def get_batch_limit(cls, plan: str) -> int:
        """Get batch size limit for a plan"""
        plan_config = cls.PLAN_LIMITS.get(plan, cls.PLAN_LIMITS["starter"])
        return plan_config["max_batch_size"]

class TokenBucket:
    """Token bucket for rate limiting"""

    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.utcnow()

    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        await self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    async def _refill(self):
        """Refill tokens based on time passed"""
        now = datetime.utcnow()
        time_passed = (now - self.last_refill).total_seconds()
        tokens_to_add = time_passed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
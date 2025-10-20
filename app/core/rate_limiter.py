"""
Rate Limiting for Aegis API
"""

import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import redis.asyncio as redis

from app.core.config import settings

class RateLimiter:
    """Distributed rate limiter using Redis"""

    def __init__(self):
        self.redis_client = None
        self.local_cache = defaultdict(lambda: {"requests": 0, "reset_time": time.time()})

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(settings.REDIS_URL)

    async def check_limit(
        self,
        customer_id: str,
        requests: int = 1,
        window: int = 60
    ) -> bool:
        """
        Check if customer is within rate limit

        Args:
            customer_id: Customer identifier
            requests: Number of requests to consume
            window: Time window in seconds

        Returns:
            True if within limit, False otherwise
        """
        if not self.redis_client:
            await self.initialize()

        key = f"rate_limit:{customer_id}"
        current_time = int(time.time())
        window_start = current_time - window

        try:
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests in window
            pipe.zcard(key)

            # Add new request
            pipe.zadd(key, {f"{current_time}:{time.time_ns()}": current_time})

            # Set expiry
            pipe.expire(key, window + 1)

            results = await pipe.execute()
            current_requests = results[1]

            # Check against limit (get from customer plan)
            limit = await self._get_customer_limit(customer_id)

            return current_requests <= limit

        except Exception as e:
            # Fallback to local rate limiting
            return self._check_local_limit(customer_id, requests, window)

    async def _get_customer_limit(self, customer_id: str) -> int:
        """Get rate limit for customer based on plan"""
        # In production, query from database
        # For now, return default based on plan
        customer_data = await self.redis_client.get(f"customer:{customer_id}")

        if customer_data:
            import json
            data = json.loads(customer_data)
            plan = data.get("plan", "starter")

            limits = {
                "starter": 100,
                "growth": 1000,
                "enterprise": 10000
            }
            return limits.get(plan, 100)

        return settings.RATE_LIMIT_REQUESTS

    def _check_local_limit(
        self,
        customer_id: str,
        requests: int,
        window: int
    ) -> bool:
        """Local rate limiting fallback"""
        current_time = time.time()
        customer_data = self.local_cache[customer_id]

        # Reset if window expired
        if current_time - customer_data["reset_time"] > window:
            customer_data["requests"] = 0
            customer_data["reset_time"] = current_time

        # Check limit
        if customer_data["requests"] + requests > settings.RATE_LIMIT_REQUESTS:
            return False

        customer_data["requests"] += requests
        return True

    async def get_remaining(self, customer_id: str, window: int = 60) -> Dict:
        """Get remaining requests for customer"""
        if not self.redis_client:
            await self.initialize()

        key = f"rate_limit:{customer_id}"
        current_time = int(time.time())
        window_start = current_time - window

        try:
            # Count current requests
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            current_requests = await self.redis_client.zcard(key)

            limit = await self._get_customer_limit(customer_id)

            return {
                "limit": limit,
                "remaining": max(0, limit - current_requests),
                "reset": current_time + window,
                "window": window
            }

        except Exception:
            return {
                "limit": settings.RATE_LIMIT_REQUESTS,
                "remaining": settings.RATE_LIMIT_REQUESTS,
                "reset": current_time + window,
                "window": window
            }

class IPRateLimiter:
    """IP-based rate limiting for additional security"""

    def __init__(self):
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(settings.REDIS_URL)

    async def check_ip_limit(
        self,
        ip_address: str,
        limit: int = 1000,
        window: int = 3600
    ) -> bool:
        """Check if IP is within rate limit"""
        if not self.redis_client:
            await self.initialize()

        key = f"ip_rate:{ip_address}"

        try:
            current = await self.redis_client.incr(key)

            if current == 1:
                await self.redis_client.expire(key, window)

            return current <= limit

        except Exception:
            return True  # Allow on error

    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        if not self.redis_client:
            await self.initialize()

        try:
            blocked = await self.redis_client.get(f"blocked_ip:{ip_address}")
            return blocked is not None
        except Exception:
            return False

    async def block_ip(self, ip_address: str, duration: int = 3600):
        """Block an IP address temporarily"""
        if not self.redis_client:
            await self.initialize()

        try:
            await self.redis_client.setex(
                f"blocked_ip:{ip_address}",
                duration,
                "blocked"
            )
        except Exception:
            pass

class CostCalculator:
    """Calculate API usage costs"""

    PRICING = {
        "starter": {
            "base_price": 4999,  # $4,999/month
            "included_requests": 10_000_000,
            "overage_per_million": 50  # $50 per million after included
        },
        "growth": {
            "base_price": 24999,  # $24,999/month
            "included_requests": 500_000_000,
            "overage_per_million": 25  # $25 per million after included
        },
        "enterprise": {
            "base_price": 75000,  # $75,000/month
            "included_requests": 10_000_000_000,
            "overage_per_million": 10  # $10 per million after included
        }
    }

    @classmethod
    def calculate_monthly_cost(cls, plan: str, requests: int) -> Dict:
        """Calculate monthly cost based on usage"""
        pricing = cls.PRICING.get(plan, cls.PRICING["starter"])

        base_cost = pricing["base_price"]
        included = pricing["included_requests"]

        if requests <= included:
            total_cost = base_cost
            overage_cost = 0
        else:
            overage = requests - included
            overage_millions = overage / 1_000_000
            overage_cost = overage_millions * pricing["overage_per_million"]
            total_cost = base_cost + overage_cost

        return {
            "base_cost": base_cost,
            "overage_cost": overage_cost,
            "total_cost": total_cost,
            "requests": requests,
            "included_requests": included
        }
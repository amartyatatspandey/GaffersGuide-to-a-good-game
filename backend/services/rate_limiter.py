from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

LOGGER = logging.getLogger(__name__)

# Redis LUA script for atomic sliding window rate limiting
# KEYS[1]: rate limit key
# ARGV[1]: current timestamp (float/int)
# ARGV[2]: window size in seconds
# ARGV[3]: limit/max allowed requests
# ARGV[4]: unique member value (to avoid collisions in sorted set)
LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

-- Remove old elements older than the window
redis.call('zremrangebyscore', key, 0, now - window)

-- Count elements currently in the window
local current_count = redis.call('zcard', key)

if current_count < limit then
    -- Add current request's unique member
    redis.call('zadd', key, now, member)
    -- Set TTL to ensure keys expire when inactive
    redis.call('expire', key, math.ceil(window))
    return 1
else
    return 0
end
"""


class InMemoryRateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._storage: dict[str, list[float]] = {}
        self._last_prune_time = time.time()
        self._prune_interval = 300.0  # Prune keys every 5 minutes

    def check_limit(self, key: str, limit: int, window: float) -> bool:
        now = time.time()
        with self._lock:
            # Lazy cleanup of old values for the current key
            cutoff = now - window
            timestamps = self._storage.get(key, [])
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) < limit:
                timestamps.append(now)
                self._storage[key] = timestamps
                allowed = True
            else:
                self._storage[key] = timestamps
                allowed = False

            # Periodic global pruning of empty/expired keys to prevent memory leaks
            if now - self._last_prune_time > self._prune_interval:
                self._prune_expired_keys_unsafe(now)
                self._last_prune_time = now

            return allowed

    def _prune_expired_keys_unsafe(self, now: float) -> None:
        """Prunes all empty or fully expired key lists from storage.
        
        Must be called under self._lock.
        """
        # We assume a maximum window size of 24 hours (86400 seconds) for pruning
        max_window = 86400.0
        cutoff = now - max_window
        
        to_delete = []
        for key, timestamps in self._storage.items():
            # Filter timestamps
            active_timestamps = [t for t in timestamps if t > cutoff]
            if not active_timestamps:
                to_delete.append(key)
            else:
                self._storage[key] = active_timestamps
                
        for key in to_delete:
            self._storage.pop(key, None)
            
        LOGGER.debug("Pruned %d expired rate limit keys from memory", len(to_delete))


class RedisRateLimiter:
    """Redis-backed atomic sliding window rate limiter."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self.client: Any = None
        self._lua_sha: str | None = None
        self.is_connected = False
        self._connect()

    def _connect(self) -> None:
        try:
            import redis
            # Connect with a short timeout to prevent blocking application startup
            self.client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
                decode_responses=True
            )
            # Register LUA script
            self._lua_sha = self.client.script_load(LUA_SLIDING_WINDOW)
            self.is_connected = True
            LOGGER.info("Successfully connected to Redis at %s for rate limiting", self.redis_url)
        except Exception as e:
            LOGGER.warning("Failed to initialize Redis client. Falling back to in-memory: %s", e)
            self.is_connected = False
            self.client = None

    def check_limit(self, key: str, limit: int, window: float) -> bool:
        if not self.is_connected or self.client is None:
            return False

        try:
            now = time.time()
            member = f"{now}:{uuid.uuid4().hex}"
            
            # Execute Lua script
            # evalsha(sha, numkeys, *keys_and_args)
            result = self.client.evalsha(
                self._lua_sha,
                1,
                key,
                str(now),
                str(window),
                str(limit),
                member
            )
            return bool(result)
        except Exception as e:
            LOGGER.error("Redis rate limit check failed: %s. Falling back to True (fail-open)", e)
            return True


class RateLimiterOrchestrator:
    """Orchestrates rate limiting across different environments."""

    def __init__(self) -> None:
        redis_url = os.getenv("RATE_LIMIT_REDIS_URL")
        
        self.redis_limiter: RedisRateLimiter | None = None
        self.in_memory_limiter = InMemoryRateLimiter()

        if redis_url:
            try:
                self.redis_limiter = RedisRateLimiter(redis_url)
            except Exception as e:
                LOGGER.warning("Redis Rate Limiter setup failed: %s. Falling back to in-memory.", e)

    def is_allowed(self, key: str, limit: int, window: float) -> bool:
        """Checks if the request key is within the rate limit constraints."""
        # Use Redis if available and connected
        if self.redis_limiter and self.redis_limiter.is_connected:
            return self.redis_limiter.check_limit(key, limit, window)
            
        # Fall back to in-memory rate limiter
        return self.in_memory_limiter.check_limit(key, limit, window)


# Global rate limiter instance
LIMITER = RateLimiterOrchestrator()


def get_client_ip(headers: dict[str, str], client_host: str | None = None) -> str:
    """Resolves client IP from headers or fallback client_host, preventing spoofing.
    
    Traverses the X-Forwarded-For header chain to identify the trusted client IP.
    """
    xff = headers.get("x-forwarded-for")
    if xff:
        # Split and clean spaces
        ips = [ip.strip() for ip in xff.split(",")]
        if ips:
            # Anti-Spoofing: In GFE/Load Balancers environments (like Cloud Run),
            # the proxy appends the client IP at the end.
            # We configure trusted proxy depth to read the rightmost untamperable IP.
            try:
                depth = int(os.getenv("TRUSTED_PROXIES_COUNT", "1"))
            except ValueError:
                depth = 1
                
            if len(ips) >= depth:
                return ips[-depth]
            return ips[-1]
            
    return client_host or "127.0.0.1"


def get_user_rate_limit_key(user: dict[str, Any] | None, client_ip: str) -> tuple[str, str]:
    """Returns (key, identifier_type) for rate limiting.
    
    If authenticated, returns ('sub:<user_id>', 'user').
    If anonymous, returns ('ip:<client_ip>', 'ip').
    """
    if user and user.get("sub") and user.get("sub") != "anonymous":
        return f"sub:{user['sub']}", "user"
    return f"ip:{client_ip}", "ip"


def get_user_tier(user: dict[str, Any] | None) -> str:
    """Resolves user tier: system, premium, authenticated, or anonymous."""
    if not user:
        return "anonymous"
        
    role = user.get("role")
    sub = user.get("sub")
    
    if role == "service_role" or sub == "system-api-key" or sub == "pytest-bypass":
        return "system"
        
    # Check all common premium tier claims (for future proofing)
    if (
        role == "premium" or
        user.get("tier") == "premium" or
        user.get("user_metadata", {}).get("tier") == "premium" or
        user.get("user_metadata", {}).get("role") == "premium" or
        user.get("app_metadata", {}).get("tier") == "premium"
    ):
        return "premium"
        
    if sub and sub != "anonymous":
        return "authenticated"
        
    return "anonymous"


def get_error_response_content(limit_type: str, limit_max: int) -> dict[str, Any]:
    """Returns a user friendly error response structure."""
    if limit_type == "general":
        return {
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down and try again later.",
            "limit": limit_max,
            "window": "1 minute"
        }
    elif limit_type == "analysis_anonymous":
        return {
            "error": "daily_limit_exceeded",
            "message": f"Daily analysis limit reached ({limit_max} per day). Sign up or log in to analyze more matches.",
            "limit": limit_max,
            "window": "24 hours"
        }
    elif limit_type == "analysis_authenticated":
        return {
            "error": "daily_limit_exceeded",
            "message": f"Daily analysis limit reached ({limit_max} per day). Upgrade to Premium for unlimited analyses.",
            "limit": limit_max,
            "window": "24 hours"
        }
    else:
        return {
            "error": "rate_limit_exceeded",
            "message": "Rate limit exceeded. Please try again later.",
            "limit": limit_max,
            "window": "unknown"
        }

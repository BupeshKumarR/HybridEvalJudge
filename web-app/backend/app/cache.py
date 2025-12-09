"""
Redis caching layer for performance optimization.
"""
import redis
import json
import logging
from typing import Optional, Any
from functools import wraps
import hashlib
import os

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default

# Initialize Redis client with connection pooling
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    max_connections=50,
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=redis_pool)


def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        str: MD5 hash of the arguments
    """
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cache_result(ttl: int = CACHE_TTL, key_prefix: str = ""):
    """
    Decorator to cache function results in Redis.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{get_cache_key(*args, **kwargs)}"
            
            try:
                # Try to get from cache
                cached_value = redis_client.get(cache_key)
                if cached_value:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return json.loads(cached_value)
                
                # Cache miss - execute function
                logger.debug(f"Cache miss for key: {cache_key}")
                result = await func(*args, **kwargs)
                
                # Store in cache
                redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
                
                return result
            except redis.RedisError as e:
                logger.warning(f"Redis error: {e}. Executing function without cache.")
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{get_cache_key(*args, **kwargs)}"
            
            try:
                # Try to get from cache
                cached_value = redis_client.get(cache_key)
                if cached_value:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return json.loads(cached_value)
                
                # Cache miss - execute function
                logger.debug(f"Cache miss for key: {cache_key}")
                result = func(*args, **kwargs)
                
                # Store in cache
                redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
                
                return result
            except redis.RedisError as e:
                logger.warning(f"Redis error: {e}. Executing function without cache.")
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def invalidate_cache(key_pattern: str):
    """
    Invalidate cache entries matching a pattern.
    
    Args:
        key_pattern: Pattern to match cache keys (supports wildcards)
    """
    try:
        keys = redis_client.keys(key_pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries matching: {key_pattern}")
    except redis.RedisError as e:
        logger.warning(f"Failed to invalidate cache: {e}")


def get_cached(key: str) -> Optional[Any]:
    """
    Get a value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except redis.RedisError as e:
        logger.warning(f"Failed to get from cache: {e}")
        return None


def set_cached(key: str, value: Any, ttl: int = CACHE_TTL):
    """
    Set a value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds
    """
    try:
        redis_client.setex(key, ttl, json.dumps(value, default=str))
    except redis.RedisError as e:
        logger.warning(f"Failed to set cache: {e}")


def check_redis_health() -> bool:
    """
    Check if Redis is available and healthy.
    
    Returns:
        bool: True if Redis is healthy, False otherwise
    """
    try:
        redis_client.ping()
        return True
    except redis.RedisError:
        return False

"""
Simple Cache Implementation

Basic caching system for immediate functionality.
"""

import asyncio
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import functools
import logging


class SimpleCache:
    """Simple in-memory cache"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.utcnow() < entry['expires']:
                self.stats['hits'] += 1
                return entry['value']
            else:
                del self.cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        expires = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expires': expires,
            'created': datetime.utcnow()
        }
        self.stats['sets'] += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.stats = {'hits': 0, 'misses': 0, 'sets': 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(1, total_requests)
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'], 
            'sets': self.stats['sets'],
            'hit_rate': hit_rate,
            'entries': len(self.cache)
        }


class SimpleCacheManager:
    """Simple cache manager"""
    
    def __init__(self):
        self.caches: Dict[str, SimpleCache] = {}
        self.default_cache = SimpleCache()
        self.caches['default'] = self.default_cache
    
    def get_cache(self, name: str = 'default') -> SimpleCache:
        """Get or create cache instance"""
        if name not in self.caches:
            self.caches[name] = SimpleCache()
        return self.caches[name]
    
    def clear_all(self) -> None:
        """Clear all caches"""
        for cache in self.caches.values():
            cache.clear()


# Global cache manager
_cache_manager = SimpleCacheManager()


def get_cache(name: str = 'default') -> SimpleCache:
    """Get cache instance"""
    return _cache_manager.get_cache(name)


def cached_quantum(cache_name: str = 'default', ttl: int = 300):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        cache = get_cache(cache_name)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator
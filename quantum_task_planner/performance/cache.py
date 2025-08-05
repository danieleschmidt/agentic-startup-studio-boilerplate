"""
Quantum Task Planner Caching System

Advanced caching implementation with quantum state preservation,
probabilistic cache invalidation, and adaptive cache optimization.
"""

import asyncio
import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from functools import wraps
from collections import OrderedDict
import threading
import weakref

import numpy as np

from ..utils.logging import get_logger


@dataclass
class CacheEntry:
    """Cache entry with quantum-aware metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    expiration: Optional[datetime]
    quantum_coherence: float = 1.0
    probability_weight: float = 1.0
    serialized_size: int = 0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.expiration is None:
            return False
        return datetime.utcnow() > self.expiration
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def quantum_decay_factor(self) -> float:
        """Calculate quantum coherence decay"""
        # Exponential decay based on age and access patterns
        age_factor = np.exp(-self.age_seconds / 3600)  # 1 hour half-life
        access_factor = min(1.0, self.access_count / 10)  # Stabilizes after 10 accesses
        return self.quantum_coherence * age_factor * access_factor


class QuantumCache:
    """
    Quantum-aware cache with probabilistic eviction and coherence-based optimization
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: int = 3600,
                 cleanup_interval: int = 300,
                 quantum_threshold: float = 0.1):
        
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.quantum_threshold = quantum_threshold
        
        # Thread-safe storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'quantum_invalidations': 0,
            'total_size': 0
        }
        
        # Background cleanup
        self._cleanup_task = None
        self._start_cleanup_task()
        
        self.logger = get_logger()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._quantum_cleanup()
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _generate_key(self, key_data: Any) -> str:
        """Generate stable cache key from data"""
        if isinstance(key_data, str):
            return key_data
        
        # Create deterministic hash from data
        if hasattr(key_data, '__dict__'):
            key_data = key_data.__dict__
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate approximate size of cache entry"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache with quantum probability consideration"""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key not in self._cache:
                self._stats['misses'] += 1
                return default
            
            entry = self._cache[cache_key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[cache_key]
                self._stats['misses'] += 1
                return default
            
            # Quantum coherence check
            if entry.quantum_decay_factor < self.quantum_threshold:
                # Probabilistic invalidation based on quantum decoherence
                if np.random.random() > entry.quantum_decay_factor:
                    del self._cache[cache_key]
                    self._stats['quantum_invalidations'] += 1
                    self._stats['misses'] += 1
                    return default
            
            # Update access metadata
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            
            self._stats['hits'] += 1
            return entry.value
    
    def set(self, key: Any, value: Any, 
            ttl: Optional[int] = None,
            quantum_coherence: float = 1.0,
            probability_weight: float = 1.0,
            dependencies: List[str] = None) -> bool:
        """Set item in cache with quantum metadata"""
        
        cache_key = self._generate_key(key)
        now = datetime.utcnow()
        
        # Calculate expiration
        expiration = None
        if ttl is not None:
            expiration = now + timedelta(seconds=ttl)
        elif self.default_ttl > 0:
            expiration = now + timedelta(seconds=self.default_ttl)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=now,
            last_accessed=now,
            access_count=0,
            expiration=expiration,
            quantum_coherence=quantum_coherence,
            probability_weight=probability_weight,
            serialized_size=self._calculate_entry_size(value),
            dependencies=dependencies or []
        )
        
        with self._lock:
            # Remove existing entry if present
            if cache_key in self._cache:
                old_entry = self._cache[cache_key]
                self._stats['total_size'] -= old_entry.serialized_size
            
            # Check size limits and evict if necessary
            self._make_space_for_entry(entry)
            
            # Add new entry
            self._cache[cache_key] = entry
            self._stats['total_size'] += entry.serialized_size
            
            return True
    
    def _make_space_for_entry(self, new_entry: CacheEntry):
        """Make space for new entry using quantum-aware eviction"""
        while len(self._cache) >= self.max_size:
            self._quantum_evict()
    
    def _quantum_evict(self):
        """Evict entry using quantum probability-weighted selection"""
        if not self._cache:
            return
        
        # Calculate eviction probabilities based on quantum factors
        candidates = []
        total_weight = 0
        
        for key, entry in self._cache.items():
            # Higher probability for older, less coherent, less accessed entries
            age_factor = entry.age_seconds / 3600  # Hours
            coherence_factor = 1.0 - entry.quantum_decay_factor
            access_factor = 1.0 / (1.0 + entry.access_count)
            size_factor = entry.serialized_size / (1024 * 1024)  # MB
            
            eviction_weight = (age_factor + coherence_factor + access_factor + size_factor) / 4
            candidates.append((key, eviction_weight))
            total_weight += eviction_weight
        
        if total_weight == 0:
            # Fallback to LRU
            key_to_evict = next(iter(self._cache))
        else:
            # Weighted random selection
            rand_val = np.random.random() * total_weight
            cumulative = 0
            key_to_evict = candidates[0][0]
            
            for key, weight in candidates:
                cumulative += weight
                if rand_val <= cumulative:
                    key_to_evict = key
                    break
        
        # Evict selected entry
        if key_to_evict in self._cache:
            evicted_entry = self._cache[key_to_evict]
            self._stats['total_size'] -= evicted_entry.serialized_size
            del self._cache[key_to_evict]
            self._stats['evictions'] += 1
    
    def invalidate(self, key: Any) -> bool:
        """Invalidate specific cache entry"""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                self._stats['total_size'] -= entry.serialized_size
                del self._cache[cache_key]
                return True
            return False
    
    def invalidate_dependencies(self, dependency_key: str):
        """Invalidate all entries that depend on a key"""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                if dependency_key in entry.dependencies:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self._cache[key]
                self._stats['total_size'] -= entry.serialized_size
                del self._cache[key]
    
    def _quantum_cleanup(self):
        """Perform quantum-aware cache cleanup"""
        with self._lock:
            now = datetime.utcnow()
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                # Remove expired entries
                if entry.is_expired:
                    keys_to_remove.append(key)
                # Remove entries with very low quantum coherence
                elif entry.quantum_decay_factor < self.quantum_threshold / 2:
                    keys_to_remove.append(key)
            
            # Remove identified entries
            for key in keys_to_remove:
                entry = self._cache[key]
                self._stats['total_size'] -= entry.serialized_size
                del self._cache[key]
            
            if keys_to_remove:
                self.logger.info(f"Quantum cleanup removed {len(keys_to_remove)} entries")
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats['total_size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = 0
            if self._stats['hits'] + self._stats['misses'] > 0:
                hit_rate = self._stats['hits'] / (self._stats['hits'] + self._stats['misses'])
            
            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_size_bytes': self._stats['total_size'],
                'avg_entry_size': self._stats['total_size'] / len(self._cache) if self._cache else 0,
                **self._stats,
                'quantum_coherence_avg': np.mean([
                    entry.quantum_decay_factor for entry in self._cache.values()
                ]) if self._cache else 0
            }


class QuantumCacheManager:
    """Global cache manager with multiple cache instances"""
    
    def __init__(self):
        self.caches: Dict[str, QuantumCache] = {}
        self._lock = threading.RLock()
    
    def get_cache(self, name: str, **kwargs) -> QuantumCache:
        """Get or create named cache instance"""
        with self._lock:
            if name not in self.caches:
                self.caches[name] = QuantumCache(**kwargs)
            return self.caches[name]
    
    def clear_all(self):
        """Clear all managed caches"""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all managed caches"""
        with self._lock:
            total_entries = sum(len(cache._cache) for cache in self.caches.values())
            total_size = sum(cache._stats['total_size'] for cache in self.caches.values())
            
            cache_stats = {}
            for name, cache in self.caches.items():
                cache_stats[name] = cache.get_stats()
            
            return {
                'total_caches': len(self.caches),
                'total_entries': total_entries,
                'total_size_bytes': total_size,
                'cache_stats': cache_stats
            }


# Global cache manager instance
_cache_manager = QuantumCacheManager()


def get_cache(name: str = "default", **kwargs) -> QuantumCache:
    """Get global cache instance"""
    return _cache_manager.get_cache(name, **kwargs)


# Caching decorators
def cached_quantum(cache_name: str = "default",
                  ttl: Optional[int] = None,
                  key_func: Optional[Callable] = None,
                  quantum_coherence: float = 1.0,
                  ignore_args: Optional[List[str]] = None):
    """
    Decorator for caching function results with quantum awareness
    """
    def decorator(func: Callable):
        cache = get_cache(cache_name)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Filter out ignored arguments
                filtered_kwargs = kwargs.copy()
                if ignore_args:
                    for arg in ignore_args:
                        filtered_kwargs.pop(arg, None)
                
                cache_key = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': filtered_kwargs
                }
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(
                cache_key, 
                result, 
                ttl=ttl,
                quantum_coherence=quantum_coherence
            )
            
            return result
        
        # Add cache management methods
        wrapper.cache_invalidate = lambda *args, **kwargs: cache.invalidate({
            'func': func.__name__,
            'args': args,
            'kwargs': kwargs
        })
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        
        return wrapper
    return decorator


def cached_quantum_async(cache_name: str = "default",
                        ttl: Optional[int] = None,
                        key_func: Optional[Callable] = None,
                        quantum_coherence: float = 1.0):
    """
    Async version of quantum caching decorator
    """
    def decorator(func: Callable):
        cache = get_cache(cache_name)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = await key_func(*args, **kwargs) if asyncio.iscoroutinefunction(key_func) else key_func(*args, **kwargs)
            else:
                cache_key = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(
                cache_key,
                result,
                ttl=ttl,
                quantum_coherence=quantum_coherence
            )
            
            return result
        
        return wrapper
    return decorator


# Specialized caches for quantum task planner
class QuantumTaskCache:
    """Specialized cache for quantum tasks"""
    
    def __init__(self):
        self.task_cache = get_cache("quantum_tasks", max_size=5000, default_ttl=1800)
        self.schedule_cache = get_cache("schedules", max_size=100, default_ttl=300)
        self.optimization_cache = get_cache("optimizations", max_size=50, default_ttl=600)
    
    def cache_task(self, task_id: str, task_data: Any, coherence: float = None):
        """Cache quantum task with coherence-based TTL"""
        ttl = int(3600 * (coherence or 1.0))  # Higher coherence = longer cache time
        self.task_cache.set(
            f"task:{task_id}",
            task_data,
            ttl=ttl,
            quantum_coherence=coherence or 1.0
        )
    
    def get_task(self, task_id: str) -> Any:
        """Get cached task"""
        return self.task_cache.get(f"task:{task_id}")
    
    def cache_schedule(self, schedule_key: str, schedule_data: Any):
        """Cache optimization schedule"""
        self.schedule_cache.set(schedule_key, schedule_data, quantum_coherence=0.8)
    
    def get_schedule(self, schedule_key: str) -> Any:
        """Get cached schedule"""
        return self.schedule_cache.get(schedule_key)
    
    def invalidate_task_dependencies(self, task_id: str):
        """Invalidate caches that depend on a specific task"""
        self.schedule_cache.clear()  # Schedules depend on task states
        self.optimization_cache.clear()  # Optimizations depend on task configurations


# Global quantum task cache
quantum_task_cache = QuantumTaskCache()
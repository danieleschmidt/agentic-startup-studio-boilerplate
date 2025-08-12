"""
Advanced Caching System

High-performance caching with distributed support, smart eviction, and quantum-aware optimization.
"""

import asyncio
import json
import time
import hashlib
import json as secure_json
import zlib
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import threading
from collections import OrderedDict, defaultdict
import heapq
import weakref

from ..utils.exceptions import CacheError
from ..utils.robust_logging import QuantumLoggerAdapter, performance_logger


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    QUANTUM = "quantum"   # Quantum-aware eviction


class CompressionType(Enum):
    """Compression types for cached data"""
    NONE = "none"
    ZLIB = "zlib"
    ADAPTIVE = "adaptive"  # Choose based on data size


@dataclass
class CacheEntry:
    """Advanced cache entry with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at
    
    def access(self):
        """Record access"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'ttl': self.ttl,
            'size_bytes': self.size_bytes,
            'compressed': self.compressed,
            'compression_type': self.compression_type.value,
            'age_seconds': self.age_seconds,
            'is_expired': self.is_expired,
            'metadata': self.metadata
        }


class CacheStats:
    """Cache statistics tracker"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compressions = 0
        self.decompressions = 0
        self.size_saved_bytes = 0
        self.start_time = time.time()
        self._lock = threading.RLock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    def record_compression(self, original_size: int, compressed_size: int):
        with self._lock:
            self.compressions += 1
            self.size_saved_bytes += (original_size - compressed_size)
    
    def record_decompression(self):
        with self._lock:
            self.decompressions += 1
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'evictions': self.evictions,
            'compressions': self.compressions,
            'decompressions': self.decompressions,
            'size_saved_mb': self.size_saved_bytes / (1024 * 1024),
            'uptime_seconds': self.uptime_seconds
        }


class QuantumAwareCache:
    """Advanced cache with quantum-aware optimization"""
    
    def __init__(
        self,
        name: str = "default",
        max_size_mb: int = 100,
        max_entries: int = 1000,
        default_ttl: Optional[float] = None,
        strategy: CacheStrategy = CacheStrategy.LRU,
        compression_threshold: int = 1024,  # Compress if larger than 1KB
        quantum_weight: float = 0.3,  # Weight for quantum coherence in eviction
        enable_persistence: bool = False
    ):
        self.name = name
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.compression_threshold = compression_threshold
        self.quantum_weight = quantum_weight
        self.enable_persistence = enable_persistence
        
        # Storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.ttl_heap = []  # For TTL-based eviction
        
        # Statistics and monitoring
        self.stats = CacheStats()
        self.current_size_bytes = 0
        self._lock = threading.RLock()
        self.logger = QuantumLoggerAdapter(
            import_logging().getLogger(f"{__name__}.{name}"),
            component=f"cache_{name}"
        )
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
            except RuntimeError:
                # No event loop running, cleanup will be manual
                pass
    
    async def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                self._evict_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
    
    def _get_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback to string representation
            return len(str(obj).encode('utf-8'))
    
    def _compress_data(self, data: Any, compression_type: CompressionType = CompressionType.ADAPTIVE) -> Tuple[Any, bool, CompressionType]:
        """Compress data if beneficial"""
        try:
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # Skip compression for small data
            if original_size < self.compression_threshold:
                return data, False, CompressionType.NONE
            
            if compression_type == CompressionType.ADAPTIVE:
                # Choose compression based on data characteristics
                compression_type = CompressionType.ZLIB
            
            if compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(serialized)
                compressed_size = len(compressed)
                
                # Only use compression if it saves significant space
                if compressed_size < original_size * 0.8:
                    self.stats.record_compression(original_size, compressed_size)
                    return compressed, True, CompressionType.ZLIB
                else:
                    return data, False, CompressionType.NONE
            
            return data, False, CompressionType.NONE
            
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return data, False, CompressionType.NONE
    
    def _decompress_data(self, data: Any, compressed: bool, compression_type: CompressionType) -> Any:
        """Decompress data if needed"""
        if not compressed:
            return data
        
        try:
            if compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(data)
                self.stats.record_decompression()
                return secure_json.loads(decompressed.decode('utf-8'))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise CacheError(f"Failed to decompress cached data: {e}")
    
    def _calculate_quantum_score(self, entry: CacheEntry) -> float:
        """Calculate quantum-aware score for eviction decisions"""
        # Base score from quantum coherence (if available)
        quantum_coherence = entry.metadata.get('quantum_coherence', 0.5)
        
        # Task priority weight
        priority_weight = entry.metadata.get('priority_weight', 0.5)
        
        # Access patterns
        recency_score = 1.0 / (1.0 + (time.time() - entry.last_accessed) / 3600)  # Hour-based decay
        frequency_score = min(1.0, entry.access_count / 10.0)  # Normalize frequency
        
        # Combine scores
        quantum_score = (
            quantum_coherence * self.quantum_weight +
            priority_weight * 0.3 +
            recency_score * 0.2 +
            frequency_score * 0.2
        )
        
        return quantum_score
    
    def _evict_by_strategy(self, count: int = 1) -> List[str]:
        """Evict entries based on strategy"""
        if not self.entries:
            return []
        
        evicted_keys = []
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            for _ in range(min(count, len(self.access_order))):
                key = next(iter(self.access_order))
                evicted_keys.append(key)
                self._remove_entry(key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_by_frequency = sorted(
                self.entries.items(),
                key=lambda x: self.access_frequency[x[0]]
            )
            for key, _ in sorted_by_frequency[:count]:
                evicted_keys.append(key)
                self._remove_entry(key)
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first
            expired_keys = [k for k, v in self.entries.items() if v.is_expired]
            for key in expired_keys[:count]:
                evicted_keys.append(key)
                self._remove_entry(key)
            
            # If not enough expired, evict oldest
            remaining = count - len(evicted_keys)
            if remaining > 0:
                sorted_by_age = sorted(
                    [(k, v) for k, v in self.entries.items() if k not in evicted_keys],
                    key=lambda x: x[1].created_at
                )
                for key, _ in sorted_by_age[:remaining]:
                    evicted_keys.append(key)
                    self._remove_entry(key)
        
        elif self.strategy == CacheStrategy.QUANTUM:
            # Evict based on quantum scores
            scored_entries = [
                (key, self._calculate_quantum_score(entry))
                for key, entry in self.entries.items()
            ]
            
            # Sort by quantum score (lower scores evicted first)
            scored_entries.sort(key=lambda x: x[1])
            
            for key, _ in scored_entries[:count]:
                evicted_keys.append(key)
                self._remove_entry(key)
        
        # Record evictions
        for _ in evicted_keys:
            self.stats.record_eviction()
        
        return evicted_keys
    
    def _evict_expired(self) -> List[str]:
        """Evict all expired entries"""
        expired_keys = []
        with self._lock:
            for key, entry in list(self.entries.items()):
                if entry.is_expired:
                    expired_keys.append(key)
                    self._remove_entry(key)
                    self.stats.record_eviction()
        
        if expired_keys:
            self.logger.info(f"Evicted {len(expired_keys)} expired entries")
        
        return expired_keys
    
    def _remove_entry(self, key: str):
        """Remove entry and update indexes"""
        if key in self.entries:
            entry = self.entries[key]
            self.current_size_bytes -= entry.size_bytes
            
            del self.entries[key]
            self.access_order.pop(key, None)
            self.access_frequency.pop(key, None)
            
            # Remove from TTL heap (lazy removal - will be filtered during pop)
    
    def _ensure_capacity(self):
        """Ensure cache doesn't exceed capacity limits"""
        # Evict expired entries first
        self._evict_expired()
        
        # Check size limit
        if self.current_size_bytes > self.max_size_bytes:
            # Evict 10% of entries or until under limit
            target_evictions = max(1, int(len(self.entries) * 0.1))
            evicted = self._evict_by_strategy(target_evictions)
            
            if evicted:
                self.logger.info(f"Evicted {len(evicted)} entries due to size limit")
        
        # Check entry count limit
        if len(self.entries) > self.max_entries:
            excess = len(self.entries) - self.max_entries
            evicted = self._evict_by_strategy(excess)
            
            if evicted:
                self.logger.info(f"Evicted {len(evicted)} entries due to count limit")
    
    @performance_logger("cache_get")
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self._lock:
            entry = self.entries.get(key)
            
            if entry is None:
                self.stats.record_miss()
                return default
            
            if entry.is_expired:
                self._remove_entry(key)
                self.stats.record_eviction()
                self.stats.record_miss()
                return default
            
            # Record access
            entry.access()
            self.access_order.move_to_end(key)
            self.access_frequency[key] += 1
            
            self.stats.record_hit()
            
            # Decompress if needed
            value = self._decompress_data(
                entry.value, entry.compressed, entry.compression_type
            )
            
            return value
    
    @performance_logger("cache_set")
    def set(self, key: str, value: Any, ttl: Optional[float] = None, 
            quantum_coherence: Optional[float] = None, priority_weight: Optional[float] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            # Prepare metadata
            metadata = {}
            if quantum_coherence is not None:
                metadata['quantum_coherence'] = quantum_coherence
            if priority_weight is not None:
                metadata['priority_weight'] = priority_weight
            
            # Compress if beneficial
            compressed_value, compressed, compression_type = self._compress_data(value)
            
            # Calculate size
            size_bytes = self._get_object_size(compressed_value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes,
                compressed=compressed,
                compression_type=compression_type,
                metadata=metadata
            )
            
            # Remove existing entry if present
            if key in self.entries:
                self._remove_entry(key)
            
            # Add new entry
            self.entries[key] = entry
            self.current_size_bytes += size_bytes
            self.access_order[key] = True
            self.access_frequency[key] = 1
            
            # Add to TTL heap if TTL is set
            if entry.ttl:
                heapq.heappush(self.ttl_heap, (entry.created_at + entry.ttl, key))
            
            # Ensure capacity limits
            self._ensure_capacity()
            
            self.logger.quantum(
                "cache_set",
                cache_name=self.name,
                key=key,
                size_bytes=size_bytes,
                compressed=compressed,
                quantum_coherence=quantum_coherence
            )
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.entries.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.ttl_heap.clear()
            self.current_size_bytes = 0
            
            self.logger.info(f"Cache {self.name} cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            entry_stats = {
                'count': len(self.entries),
                'size_mb': self.current_size_bytes / (1024 * 1024),
                'avg_size_bytes': self.current_size_bytes / max(1, len(self.entries)),
                'compressed_count': sum(1 for e in self.entries.values() if e.compressed),
                'expired_count': sum(1 for e in self.entries.values() if e.is_expired)
            }
            
            return {
                'name': self.name,
                'strategy': self.strategy.value,
                'entries': entry_stats,
                'limits': {
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'max_entries': self.max_entries,
                    'utilization_pct': (self.current_size_bytes / self.max_size_bytes) * 100
                },
                **self.stats.to_dict()
            }
    
    def get_key_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about specific key"""
        with self._lock:
            entry = self.entries.get(key)
            return entry.to_dict() if entry else None
    
    async def shutdown(self):
        """Shutdown cache and cleanup resources"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.enable_persistence:
            await self._persist_to_disk()
        
        self.clear()
    
    async def _persist_to_disk(self):
        """Persist cache to disk (placeholder for future implementation)"""
        # TODO: Implement disk persistence
        pass


class DistributedCacheManager:
    """Manage multiple cache instances with distributed capabilities"""
    
    def __init__(self):
        self.caches: Dict[str, QuantumAwareCache] = {}
        self.default_cache = QuantumAwareCache("default")
        self.caches["default"] = self.default_cache
        self._lock = threading.RLock()
        
        self.logger = QuantumLoggerAdapter(
            import_logging().getLogger(__name__),
            component="cache_manager"
        )
    
    def get_cache(self, name: str, **kwargs) -> QuantumAwareCache:
        """Get or create cache instance"""
        with self._lock:
            if name not in self.caches:
                self.caches[name] = QuantumAwareCache(name=name, **kwargs)
                self.logger.info(f"Created cache: {name}")
            
            return self.caches[name]
    
    def delete_cache(self, name: str) -> bool:
        """Delete cache instance"""
        if name == "default":
            return False  # Cannot delete default cache
        
        with self._lock:
            if name in self.caches:
                cache = self.caches.pop(name)
                asyncio.create_task(cache.shutdown())
                self.logger.info(f"Deleted cache: {name}")
                return True
            return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches"""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self.caches.items()}
    
    def clear_all(self):
        """Clear all caches"""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
            self.logger.info("All caches cleared")
    
    async def shutdown_all(self):
        """Shutdown all caches"""
        with self._lock:
            shutdown_tasks = [cache.shutdown() for cache in self.caches.values()]
            await asyncio.gather(*shutdown_tasks)
            self.caches.clear()
            self.logger.info("All caches shutdown")


# Global cache manager
_cache_manager = DistributedCacheManager()


def get_cache(name: str = "default", **kwargs) -> QuantumAwareCache:
    """Get cache instance from global manager"""
    return _cache_manager.get_cache(name, **kwargs)


def get_cache_manager() -> DistributedCacheManager:
    """Get global cache manager"""
    return _cache_manager


def quantum_cached(
    cache_name: str = "default",
    ttl: Optional[float] = 300,
    key_func: Optional[Callable] = None,
    include_quantum_context: bool = True
):
    """Decorator for quantum-aware caching"""
    def decorator(func: Callable):
        cache = get_cache(cache_name)
        
        def generate_key(*args, **kwargs) -> str:
            if key_func:
                return key_func(*args, **kwargs)
            
            # Generate key from function and arguments
            key_data = {
                'function': f"{func.__module__}.{func.__name__}",
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            
            if include_quantum_context:
                # Add quantum context to key if available
                try:
                    from ..utils.robust_logging import correlation_id
                    key_data['correlation_id'] = correlation_id.get()
                except (ImportError, LookupError):
                    pass
            
            key_str = json.dumps(key_data, sort_keys=True)
            return hashlib.sha256(key_str.encode(), usedforsecurity=False).hexdigest()[:32]
        
        def sync_wrapper(*args, **kwargs):
            key = generate_key(*args, **kwargs)
            
            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Extract quantum context for caching
            quantum_coherence = None
            priority_weight = None
            
            if include_quantum_context and hasattr(result, 'quantum_coherence'):
                quantum_coherence = result.quantum_coherence
            
            if include_quantum_context and hasattr(result, 'priority'):
                priority_weight = getattr(result.priority, 'probability_weight', None)
            
            # Cache result
            cache.set(key, result, ttl, quantum_coherence, priority_weight)
            
            return result
        
        async def async_wrapper(*args, **kwargs):
            key = generate_key(*args, **kwargs)
            
            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Extract quantum context for caching
            quantum_coherence = None
            priority_weight = None
            
            if include_quantum_context and hasattr(result, 'quantum_coherence'):
                quantum_coherence = result.quantum_coherence
            
            if include_quantum_context and hasattr(result, 'priority'):
                priority_weight = getattr(result.priority, 'probability_weight', None)
            
            # Cache result
            cache.set(key, result, ttl, quantum_coherence, priority_weight)
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def import_logging():
    """Import logging module"""
    import logging
    return logging
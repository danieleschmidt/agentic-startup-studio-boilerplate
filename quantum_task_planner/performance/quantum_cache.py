"""
Quantum-Enhanced Caching System for Generation 3 Performance Optimization
"""

import time
import hashlib
import threading
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with quantum-inspired properties"""
    value: T
    timestamp: datetime
    access_count: int = 0
    probability_weight: float = 1.0
    quantum_coherence: float = 1.0
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry is expired"""
        if ttl_seconds <= 0:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > ttl_seconds
    
    def decay_coherence(self, decay_rate: float = 0.01):
        """Apply quantum coherence decay over time"""
        time_factor = (datetime.now() - self.timestamp).total_seconds() / 3600  # hours
        self.quantum_coherence = max(0.1, self.quantum_coherence - (decay_rate * time_factor))
    
    def boost_weight(self, boost_factor: float = 0.1):
        """Boost probability weight on access"""
        self.access_count += 1
        self.probability_weight = min(2.0, self.probability_weight + boost_factor)


class QuantumCache:
    """Quantum-enhanced cache with adaptive behavior"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: float = 3600,  # 1 hour
                 coherence_threshold: float = 0.3):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.coherence_threshold = coherence_threshold
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments"""
        # Create a deterministic key from function name and arguments
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Convert args and kwargs to string
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Create hash
        key_content = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with quantum behavior"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired(self.default_ttl):
                    del self._cache[key]
                    self._misses += 1
                    return None
                
                # Check quantum coherence
                entry.decay_coherence()
                if entry.quantum_coherence < self.coherence_threshold:
                    # Low coherence - probabilistic return
                    import random
                    if random.random() > entry.quantum_coherence:
                        self._misses += 1
                        return None
                
                # Successful hit
                entry.boost_weight()
                self._hits += 1
                return entry.value
            
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store value in cache with quantum properties"""
        with self._lock:
            # Check if eviction is needed
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_quantum_least_useful()
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=datetime.now(),
                probability_weight=1.0,
                quantum_coherence=1.0
            )
            
            self._cache[key] = entry
    
    def _evict_quantum_least_useful(self) -> None:
        """Evict least useful entry based on quantum scoring"""
        if not self._cache:
            return
        
        # Calculate quantum usefulness score for each entry
        scored_entries = []
        current_time = datetime.now()
        
        for key, entry in self._cache.items():
            # Factors: recency, access frequency, probability weight, coherence
            age_hours = (current_time - entry.timestamp).total_seconds() / 3600
            recency_score = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
            frequency_score = min(1.0, entry.access_count / 10)  # Normalize access count
            
            usefulness_score = (
                recency_score * 0.3 + 
                frequency_score * 0.3 + 
                entry.probability_weight * 0.2 + 
                entry.quantum_coherence * 0.2
            )
            
            scored_entries.append((usefulness_score, key))
        
        # Remove least useful entry
        scored_entries.sort()
        least_useful_key = scored_entries[0][1]
        del self._cache[least_useful_key]
        self._evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "average_coherence": self._calculate_average_coherence()
            }
    
    def _calculate_average_coherence(self) -> float:
        """Calculate average quantum coherence of cached entries"""
        if not self._cache:
            return 0.0
        
        total_coherence = sum(entry.quantum_coherence for entry in self._cache.values())
        return total_coherence / len(self._cache)


# Global cache instance
quantum_cache = QuantumCache()


def cache_quantum_result(ttl: Optional[float] = None, 
                        cache_instance: Optional[QuantumCache] = None):
    """Decorator to cache function results with quantum behavior"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = cache_instance or quantum_cache
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            key = cache._generate_key(func, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result
        
        return wrapper
    return decorator


class AdaptiveCache:
    """Adaptive cache that learns from usage patterns"""
    
    def __init__(self, base_cache: QuantumCache):
        self.base_cache = base_cache
        self.access_patterns: Dict[str, list] = {}
        self.learning_enabled = True
    
    def record_access_pattern(self, key: str):
        """Record access pattern for learning"""
        if not self.learning_enabled:
            return
        
        current_time = time.time()
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        
        # Keep only recent history (last 100 accesses)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def predict_next_access(self, key: str) -> Optional[float]:
        """Predict when key will be accessed next"""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 2:
            return None
        
        accesses = self.access_patterns[key]
        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
        
        # Simple prediction: average interval
        avg_interval = sum(intervals) / len(intervals)
        last_access = accesses[-1]
        
        return last_access + avg_interval
    
    def get_adaptive_ttl(self, key: str, default_ttl: float) -> float:
        """Calculate adaptive TTL based on access patterns"""
        predicted_next = self.predict_next_access(key)
        if predicted_next is None:
            return default_ttl
        
        current_time = time.time()
        predicted_interval = predicted_next - current_time
        
        # Adjust TTL based on prediction
        if predicted_interval > 0:
            # Extend TTL if we expect access soon
            return max(default_ttl, predicted_interval * 1.2)
        else:
            # Use default TTL
            return default_ttl


class QuantumCacheCluster:
    """Distributed quantum cache cluster for scaling"""
    
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.hash_ring = self._build_hash_ring()
    
    def _build_hash_ring(self) -> Dict[int, QuantumCache]:
        """Build consistent hash ring for cache distribution"""
        ring = {}
        for i, node in enumerate(self.nodes):
            # Create multiple virtual nodes for better distribution
            for v in range(3):
                hash_key = hashlib.md5(f"{i}:{v}".encode()).hexdigest()
                ring[int(hash_key[:8], 16)] = node
        
        return dict(sorted(ring.items()))
    
    def _get_node_for_key(self, key: str) -> QuantumCache:
        """Get cache node responsible for given key"""
        key_hash = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        
        # Find first node with hash >= key_hash
        for node_hash, node in self.hash_ring.items():
            if node_hash >= key_hash:
                return node
        
        # Wrap around to first node
        return list(self.hash_ring.values())[0]
    
    def get(self, key: str) -> Optional[Any]:
        """Get from appropriate node in cluster"""
        node = self._get_node_for_key(key)
        return node.get(key)
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put to appropriate node in cluster"""
        node = self._get_node_for_key(key)
        node.put(key, value, ttl)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics for entire cluster"""
        total_stats = {
            "nodes": len(self.nodes),
            "total_size": 0,
            "total_hits": 0,
            "total_misses": 0,
            "total_evictions": 0,
            "node_stats": []
        }
        
        for i, node in enumerate(self.nodes):
            stats = node.get_stats()
            total_stats["total_size"] += stats["size"]
            total_stats["total_hits"] += stats["hits"]
            total_stats["total_misses"] += stats["misses"]
            total_stats["total_evictions"] += stats["evictions"]
            total_stats["node_stats"].append({"node_id": i, **stats})
        
        total_requests = total_stats["total_hits"] + total_stats["total_misses"]
        total_stats["cluster_hit_rate"] = (
            total_stats["total_hits"] / total_requests if total_requests > 0 else 0
        )
        
        return total_stats
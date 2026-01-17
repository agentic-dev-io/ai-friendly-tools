#!/usr/bin/env python3
"""Caching system for web tool results."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default
    cache_dir: str = "./data/cache"
    max_cache_size_mb: int = 100


class CacheEntry(BaseModel):
    """A single cache entry."""

    key: str
    value: Any
    timestamp: float
    ttl: int


class Cache:
    """Simple file-based cache for web tool results."""

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache."""
        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache initialized at {self.cache_dir}")

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides config default)
        """
        if not self.config.enabled:
            return

        ttl = ttl or self.config.ttl_seconds
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl,
        )

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w") as f:
                json.dump(entry.model_dump(), f)
            logger.debug(f"Cached key: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache key {key}: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.config.enabled:
            return None

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                entry = CacheEntry(**data)

            # Check if expired
            age = time.time() - entry.timestamp
            if age > entry.ttl:
                logger.debug(f"Cache expired for key: {key}")
                cache_path.unlink()  # Delete expired entry
                return None

            logger.debug(f"Cache hit for key: {key}")
            return entry.value
        except Exception as e:
            logger.warning(f"Failed to retrieve cache for key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"Deleted cache for key: {key}")
                return True
            except Exception as e:
                logger.warning(f"Failed to delete cache for key {key}: {e}")
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_size(self) -> int:
        """Get cache size in MB.

        Returns:
            Cache size in MB
        """
        total_size = 0
        for file in self.cache_dir.glob("*.json"):
            total_size += file.stat().st_size
        return total_size // (1024 * 1024)

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                        entry = CacheEntry(**data)

                    age = time.time() - entry.timestamp
                    if age > entry.ttl:
                        cache_file.unlink()
                        count += 1
                except Exception as e:
                    logger.debug(f"Error cleaning cache file {cache_file}: {e}")

            logger.info(f"Cleaned {count} expired cache entries")
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            return 0

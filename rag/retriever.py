"""
CTrackAI — Device Spec Retriever (Stage 3 — Production Grade)

High-level retrieval interface for the pipeline. Wraps the vector
store with a thread-safe LRU cache.

Production Features:
    - Thread-safe cache (threading.Lock)
    - Similarity score passthrough to pipeline
    - Graceful degradation if RAG unavailable
    - Auto-seeds vector store on first retrieval
    - Dynamic device addition API
"""

import threading
from typing import Optional, Dict, Tuple
from loguru import logger

from models.schemas import DeviceSpec
from rag.vector_store import (
    get_device_spec_as_schema,
    search_device_specs,
    seed_vector_store,
    add_device,
    get_store_stats,
    is_available,
)


class DeviceSpecRetriever:
    """
    Thread-safe cached device specification retriever.

    Usage:
        retriever = DeviceSpecRetriever()
        spec, similarity = retriever.get("Dell OptiPlex 3090")
        if spec:
            print(f"Found: {spec.device_name} (sim={similarity})")
    """

    def __init__(self, cache_size: int = 256):
        self.cache_size = cache_size
        self._cache: Dict[str, Tuple[Optional[DeviceSpec], float]] = {}
        self._lock = threading.Lock()  # Fix #2: Thread-safe cache
        self._seeded = False
        self._stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "not_found": 0,
            "rag_unavailable": 0,
        }
        logger.info(f"DeviceSpecRetriever initialized (cache_size={cache_size})")

    def _ensure_seeded(self) -> None:
        """Seed vector store on first use (thread-safe)."""
        if self._seeded:
            return

        with self._lock:
            if self._seeded:
                return
            try:
                if not is_available():
                    logger.warning("RAG layer unavailable — skipping seed")
                    self._seeded = True
                    return

                store_stats = get_store_stats()
                if store_stats.get("total_devices", 0) == 0:
                    logger.info("Vector store empty — seeding...")
                    count = seed_vector_store()
                    logger.info(f"Seeded {count} devices")
                else:
                    logger.info(
                        f"Vector store has {store_stats['total_devices']} devices"
                    )
                self._seeded = True
            except Exception as e:
                logger.error(f"Failed to seed: {e}")
                self._seeded = True  # Don't retry

    def get(
        self,
        device_query: str,
        category_hint: Optional[str] = None,
    ) -> Tuple[Optional[DeviceSpec], float]:
        """
        Get device specifications for a given device query.
        (Fix #5: Returns (DeviceSpec, similarity_score) tuple)

        Lookup strategy:
            1. Check thread-safe cache
            2. Semantic search in ChromaDB
            3. Category-based fallback
            4. Return (None, 0.0)

        Args:
            device_query: Device name, description, or identifier
            category_hint: Optional category for fallback

        Returns:
            (DeviceSpec, similarity_score) — DeviceSpec is None if not found
        """
        self._ensure_seeded()

        # Fix #3: Graceful degradation if RAG unavailable
        if not is_available():
            self._stats["rag_unavailable"] += 1
            return None, 0.0

        self._stats["total_lookups"] += 1
        cache_key = f"{device_query}|{category_hint or ''}"

        # Fix #2: Thread-safe cache read
        with self._lock:
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                return self._cache[cache_key]

        self._stats["cache_misses"] += 1

        # Primary: semantic search
        spec, similarity = get_device_spec_as_schema(
            query=device_query,
            category_filter=category_hint,
        )

        # Fallback: category-only search
        if spec is None and category_hint:
            spec, similarity = get_device_spec_as_schema(
                query=f"generic {category_hint}",
                category_filter=category_hint,
            )

        if spec is None:
            self._stats["not_found"] += 1
            logger.debug(f"No device spec found for: '{device_query}'")

        # Fix #2: Thread-safe cache write
        with self._lock:
            if len(self._cache) >= self.cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = (spec, similarity)

        return spec, similarity

    def add_device_spec(self, device: dict) -> bool:
        """
        Add a new device to the vector store at runtime.
        (Fix #4: Dynamic device addition)

        Also invalidates any cache entries that might match.

        Args:
            device: Device dict with device_name, category, etc.

        Returns:
            True if added successfully
        """
        success = add_device(device)
        if success:
            # Invalidate related cache entries
            self.invalidate_cache(device.get("device_name", ""))
        return success

    def get_multiple(
        self,
        queries: list[str],
    ) -> Dict[str, Tuple[Optional[DeviceSpec], float]]:
        """Batch lookup for multiple device queries."""
        return {query: self.get(query) for query in queries}

    def invalidate_cache(self, device_query: str = None) -> None:
        """Invalidate cache entries (thread-safe)."""
        with self._lock:
            if device_query:
                keys = [k for k in self._cache if device_query.lower() in k.lower()]
                for key in keys:
                    del self._cache[key]
                logger.info(f"Invalidated {len(keys)} cache entries for '{device_query}'")
            else:
                self._cache.clear()
                logger.info("Cleared entire device spec cache")

    def get_retriever_stats(self) -> dict:
        """Get retriever stats including cache performance."""
        total = self._stats["total_lookups"]
        hit_rate = round(self._stats["cache_hits"] / total, 4) if total > 0 else 0
        with self._lock:
            cache_size = len(self._cache)
        return {
            **self._stats,
            "cache_hit_rate": hit_rate,
            "cache_size": cache_size,
            "max_cache_size": self.cache_size,
            "rag_available": is_available(),
            "vector_store": get_store_stats(),
        }


# ── Module-level singleton ────────────────────────────────────────
device_retriever = DeviceSpecRetriever()

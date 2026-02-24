"""
CTrackAI — Vector Store Manager (Stage 3 — Production Grade)

Manages the ChromaDB vector database for device specifications.
Uses LangChain + sentence-transformers for embedding and retrieval.

Production Features:
    - Similarity threshold (rejects low-quality matches)
    - Graceful embedding model fallback
    - Dynamic device addition at runtime
    - Health check endpoint
    - Idempotent batch seeding
"""

import json
import os
import threading
from pathlib import Path
from typing import List, Optional

from loguru import logger

from models.schemas import DeviceSpec
from config.settings import settings


# ── Thread-safe lazy singletons ───────────────────────────────────
_init_lock = threading.Lock()
_chroma_client = None
_collection = None
_embedding_function = None
_embedding_available = True  # Set to False if model fails to load


def _get_embedding_function():
    """
    Lazily load the sentence-transformer embedding function.
    Thread-safe via lock. Falls back gracefully if model unavailable.

    Returns:
        HuggingFaceEmbeddings instance or None if unavailable
    """
    global _embedding_function, _embedding_available

    if not _embedding_available:
        return None

    if _embedding_function is not None:
        return _embedding_function

    with _init_lock:
        # Double-check after acquiring lock
        if _embedding_function is not None:
            return _embedding_function

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embedding_function = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            _embedding_available = False
            logger.error(
                f"Failed to load embedding model '{settings.EMBEDDING_MODEL}': {e}. "
                f"RAG layer will be unavailable — pipeline will use default device specs."
            )
            return None

    return _embedding_function


def _get_chroma_collection():
    """
    Get or create the ChromaDB collection. Thread-safe.

    Returns:
        (chromadb.Client, chromadb.Collection) tuple or (None, None) on failure
    """
    global _chroma_client, _collection

    if _collection is not None:
        return _chroma_client, _collection

    with _init_lock:
        if _collection is not None:
            return _chroma_client, _collection

        try:
            import chromadb

            persist_dir = settings.CHROMA_PERSIST_DIR
            os.makedirs(persist_dir, exist_ok=True)

            _chroma_client = chromadb.PersistentClient(path=persist_dir)
            _collection = _chroma_client.get_or_create_collection(
                name="device_specs",
                metadata={"description": "IoT device specifications for carbon tracking"},
            )

            logger.info(
                f"ChromaDB collection 'device_specs' ready at {persist_dir} "
                f"({_collection.count()} entries)"
            )
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            return None, None

    return _chroma_client, _collection


def _device_to_document(device: dict) -> str:
    """Convert a device spec dict to a searchable text document."""
    parts = [
        device["device_name"],
        f"category: {device['category']}",
        f"rated power: {device['rated_wattage']}W",
        f"idle power: {device['idle_wattage']}W",
        f"typical power: {device['typical_wattage']}W",
        f"duty cycle: {device.get('duty_cycle', 0) * 100:.0f}%",
        f"usage hours: {device.get('typical_usage_hours', 0)}h/day",
    ]
    if device.get("energy_star_rated"):
        parts.append("Energy Star certified")
    return " | ".join(parts)


def _device_to_id(device: dict) -> str:
    """Generate a stable ID for a device spec."""
    name = device["device_name"].lower().replace(" ", "_").replace("/", "_")
    return f"device_{name}"


def is_available() -> bool:
    """
    Check if the RAG layer is available (embedding model + ChromaDB).
    Used by the pipeline for graceful degradation.

    Returns:
        True if RAG is fully operational
    """
    if not _embedding_available:
        return False
    embeddings = _get_embedding_function()
    client, collection = _get_chroma_collection()
    return embeddings is not None and collection is not None


def seed_vector_store(
    seed_file: str = None,
    force_reseed: bool = False,
) -> int:
    """
    Seed the ChromaDB vector store with device specifications.

    Args:
        seed_file: Path to seed data JSON (default: rag/device_seed_data.json)
        force_reseed: If True, clear and re-seed everything

    Returns:
        Number of devices seeded, or -1 on failure
    """
    if seed_file is None:
        seed_file = str(Path(__file__).parent / "device_seed_data.json")

    with open(seed_file, "r") as f:
        devices = json.load(f)

    logger.info(f"Loading {len(devices)} devices from {seed_file}")

    client, collection = _get_chroma_collection()
    embeddings = _get_embedding_function()

    if collection is None or embeddings is None:
        logger.error("Cannot seed: ChromaDB or embedding model unavailable")
        return -1

    if force_reseed and collection.count() > 0:
        logger.warning("Force reseed: clearing existing device specs")
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    documents = []
    metadatas = []
    ids = []

    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    for device in devices:
        device_id = _device_to_id(device)
        if device_id in existing_ids and not force_reseed:
            continue

        documents.append(_device_to_document(device))
        metadatas.append({
            "device_name": device["device_name"],
            "category": device["category"],
            "rated_wattage": float(device["rated_wattage"]),
            "idle_wattage": float(device["idle_wattage"]),
            "typical_wattage": float(device["typical_wattage"]),
            "duty_cycle": float(device.get("duty_cycle", 0)),
            "typical_usage_hours": float(device.get("typical_usage_hours", 0)),
            "energy_star_rated": device.get("energy_star_rated", False),
        })
        ids.append(device_id)

    if not documents:
        logger.info("No new devices to seed (all already exist)")
        return 0

    # Batch insert
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_docs)

        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=batch_embeddings,
        )

    logger.info(f"Seeded {len(documents)} devices (total: {collection.count()})")
    return len(documents)


def add_device(device: dict) -> bool:
    """
    Add a single device to the vector store at runtime.
    (Fix #4: Dynamic device addition)

    Args:
        device: Device dict with keys: device_name, category,
                rated_wattage, idle_wattage, typical_wattage, etc.

    Returns:
        True if added successfully, False on failure
    """
    client, collection = _get_chroma_collection()
    embeddings = _get_embedding_function()

    if collection is None or embeddings is None:
        logger.error("Cannot add device: RAG layer unavailable")
        return False

    try:
        device_id = _device_to_id(device)

        # Check if already exists
        existing = collection.get(ids=[device_id])
        if existing["ids"]:
            logger.info(f"Device '{device['device_name']}' already exists, updating")
            collection.delete(ids=[device_id])

        doc = _device_to_document(device)
        embedding = embeddings.embed_documents([doc])

        collection.add(
            documents=[doc],
            metadatas=[{
                "device_name": device["device_name"],
                "category": device["category"],
                "rated_wattage": float(device["rated_wattage"]),
                "idle_wattage": float(device["idle_wattage"]),
                "typical_wattage": float(device.get("typical_wattage", 0)),
                "duty_cycle": float(device.get("duty_cycle", 0)),
                "typical_usage_hours": float(device.get("typical_usage_hours", 0)),
                "energy_star_rated": device.get("energy_star_rated", False),
            }],
            ids=[device_id],
            embeddings=embedding,
        )

        logger.info(f"Added device: {device['device_name']} (id: {device_id})")
        return True

    except Exception as e:
        logger.error(f"Failed to add device '{device.get('device_name', '?')}': {e}")
        return False


def search_device_specs(
    query: str,
    top_k: int = 3,
    category_filter: Optional[str] = None,
    similarity_threshold: float = None,
) -> List[dict]:
    """
    Search for device specifications by query text.
    (Fix #1: Similarity threshold filtering)

    Args:
        query: Search query (device name, category, or description)
        top_k: Number of results to return
        category_filter: Optional category filter
        similarity_threshold: Min similarity to accept (default: from config)

    Returns:
        List of matching device specs with similarity scores,
        filtered by threshold
    """
    if similarity_threshold is None:
        similarity_threshold = settings.RAG_SIMILARITY_THRESHOLD

    client, collection = _get_chroma_collection()
    embeddings = _get_embedding_function()

    if collection is None or embeddings is None:
        logger.warning("RAG layer unavailable — returning empty results")
        return []

    if collection.count() == 0:
        logger.warning("Vector store is empty. Run seed_vector_store() first.")
        return []

    query_embedding = embeddings.embed_query(query)

    where_filter = None
    if category_filter:
        where_filter = {"category": category_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 / (1.0 + distance)

            # Fix #1: Skip results below threshold
            if similarity < similarity_threshold:
                logger.debug(
                    f"Skipping '{metadata['device_name']}' — "
                    f"similarity {similarity:.4f} < threshold {similarity_threshold}"
                )
                continue

            formatted.append({
                "id": doc_id,
                "similarity": round(similarity, 4),
                "device_name": metadata["device_name"],
                "category": metadata["category"],
                "rated_wattage": metadata["rated_wattage"],
                "idle_wattage": metadata["idle_wattage"],
                "typical_wattage": metadata["typical_wattage"],
                "duty_cycle": metadata["duty_cycle"],
                "typical_usage_hours": metadata["typical_usage_hours"],
                "energy_star_rated": metadata["energy_star_rated"],
            })

    return formatted


def get_device_spec_as_schema(
    query: str,
    category_filter: Optional[str] = None,
) -> Optional[tuple]:
    """
    Search for a device and return the top match as (DeviceSpec, similarity).
    (Fix #5: Returns similarity score alongside DeviceSpec)

    Args:
        query: Device name or description
        category_filter: Optional category filter

    Returns:
        (DeviceSpec, similarity_score) if found, (None, 0.0) if no match
    """
    results = search_device_specs(query, top_k=1, category_filter=category_filter)

    if not results:
        return None, 0.0

    top = results[0]
    spec = DeviceSpec(
        device_name=top["device_name"],
        category=top["category"],
        rated_wattage=top["rated_wattage"],
        idle_wattage=top["idle_wattage"],
        typical_wattage=top.get("typical_wattage"),
        duty_cycle=top.get("duty_cycle"),
        typical_usage_hours=top.get("typical_usage_hours"),
        energy_star_rated=top.get("energy_star_rated", False),
    )

    return spec, top["similarity"]


def get_store_stats() -> dict:
    """Get current vector store statistics and health."""
    try:
        client, collection = _get_chroma_collection()
        if collection is None:
            return {"status": "unavailable", "error": "ChromaDB not initialized"}
        return {
            "status": "healthy",
            "total_devices": collection.count(),
            "persist_dir": settings.CHROMA_PERSIST_DIR,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_available": _embedding_available,
            "similarity_threshold": settings.RAG_SIMILARITY_THRESHOLD,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

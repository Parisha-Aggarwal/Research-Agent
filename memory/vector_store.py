"""
memory/vector_store.py — ChromaDB wrapper for storing and searching findings.

How it works:
  1. sentence-transformers converts text → a 384-dim vector (list of floats)
  2. ChromaDB stores that vector alongside the original text on disk
  3. On search, your query is also converted to a vector
  4. ChromaDB finds stored vectors that are mathematically "close" to it
  5. Close vectors = semantically similar content

This means searching "power usage" will surface findings about
"energy consumption" — even with zero shared words.
"""

import logging
import hashlib
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages storage and retrieval of research findings using ChromaDB.

    Usage:
        store = VectorStore()
        store.add_finding(url="https://...", content="...", sub_question="...")
        results = store.search_findings("energy efficiency", n_results=3)
    """

    def __init__(self):
        """
        Set up ChromaDB client and load the embedding model.
        Both are initialized once and reused — loading is slow,
        so we do it at startup, not on every call.
        """
        logger.info("Initializing VectorStore...")

        # PersistentClient saves vectors to disk between sessions
        # Settings(anonymized_telemetry=False) stops ChromaDB phoning home
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create_collection is safe to call on every startup —
        # it reuses an existing collection or creates a fresh one
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

        # Load embedding model (downloads ~80 MB on first run, then cached)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("VectorStore ready")

    def add_finding(self, url: str, content: str, sub_question: str) -> bool:
        """
        Embed and store one research finding.

        Args:
            url:          Source URL (used as part of the unique ID)
            content:      Scraped text content to store and embed
            sub_question: Which research sub-question this answers

        Returns:
            True on success, False on failure.

        Why we hash the ID:
            ChromaDB requires unique string IDs. Using a hash of
            (url + sub_question) means re-adding the same finding
            is a no-op rather than a duplicate.
        """
        if not content.strip():
            logger.warning(f"Skipping empty content for {url}")
            return False

        try:
            # Create a stable unique ID from url + sub_question
            raw_id = f"{url}::{sub_question}"
            doc_id = hashlib.md5(raw_id.encode()).hexdigest()

            # Convert text to a 384-dim vector
            embedding = self._embedder.encode(content).tolist()

            # Store: vector + original text + metadata
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "url":          url,
                    "sub_question": sub_question,
                }],
            )

            logger.info(f"Stored finding from: {url[:60]}")
            return True

        except Exception as e:
            logger.error(f"Failed to store finding from {url}: {e}")
            return False

    def search_findings(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Retrieve the most semantically relevant findings for a query.

        Args:
            query:     Natural language search query.
            n_results: How many results to return.

        Returns:
            List of dicts, each with: content, url, sub_question, score.
            Score is cosine distance (lower = more similar, 0.0 = identical).
            Returns empty list on failure.
        """
        try:
            # Don't query if the collection is empty
            count = self._collection.count()
            if count == 0:
                logger.warning("Collection is empty — nothing to search")
                return []

            # Cap n_results to what's actually stored
            n_results = min(n_results, count)

            # Embed the query using the same model as the stored documents
            query_embedding = self._embedder.encode(query).tolist()

            raw = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            # Unpack ChromaDB's nested list format into flat dicts
            results = []
            documents = raw["documents"][0]
            metadatas = raw["metadatas"][0]
            distances = raw["distances"][0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                results.append({
                    "content":      doc,
                    "url":          meta.get("url", ""),
                    "sub_question": meta.get("sub_question", ""),
                    "score":        round(dist, 4),  # cosine distance
                })

            logger.info(f"Found {len(results)} relevant findings for: '{query[:50]}'")
            return results

        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []

    def clear_collection(self) -> bool:
        """
        Delete and recreate the collection — wipes all stored findings.
        Call this at the start of each new research session.

        Returns:
            True on success, False on failure.
        """
        try:
            self._client.delete_collection(CHROMA_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection cleared for new session")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def count(self) -> int:
        """Return how many findings are currently stored."""
        return self._collection.count()
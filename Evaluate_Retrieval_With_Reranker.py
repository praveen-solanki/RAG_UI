"""
ADVANCED HYBRID RETRIEVAL SYSTEM
==================================
Features:
- Hybrid search (Dense + Sparse BM25)
- Metadata filtering
- Query expansion
- Result caching
- Thread-safe parallel search
- CrossEncoder reranker (BAAI/bge-reranker-large on CUDA)
- Multi-server Ollama support (gen vs judge)
- SentenceTransformer BAAI/bge-m3 on CUDA for embeddings
"""

import json
import time
import hashlib
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from rank_bm25 import BM25Okapi

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SparseVector,
)
from sklearn.metrics.pairwise import cosine_similarity
import requests

try:
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    USE_NLTK = True
except ImportError:
    USE_NLTK = False

# ================= CONFIG =================

COLLECTION  = "rag_hybrid_bge_m3"
QDRANT_URL  = "http://localhost:7333"

# Embedding
BGE_MODEL   = "BAAI/bge-m3"

# Reranker
RERANKER_MODEL      = "BAAI/bge-reranker-large"
USE_CROSS_ENCODER   = True
USE_OLLAMA_RERANKER = False

# Multi-server Ollama
OLLAMA_GEN_URL   = "http://localhost:11435"
OLLAMA_JUDGE_URL = "http://localhost:11434"
OLLAMA_BASE_URL  = OLLAMA_GEN_URL

# Retrieval parameters
DENSE_TOP_K  = 20
SPARSE_TOP_K = 20
HYBRID_TOP_K = 20
FINAL_TOP_K  = 8

# Fusion weights
DENSE_WEIGHT  = 0.7
SPARSE_WEIGHT = 0.3

# Query expansion
ENABLE_QUERY_EXPANSION = True
EXPANSION_SYNONYMS = {
    "error":    ["failure", "fault", "issue", "problem", "exception"],
    "config":   ["configuration", "settings", "setup", "parameters"],
    "install":  ["installation", "setup", "deployment"],
    "api":      ["interface", "endpoint", "service", "rest"],
}

# Caching
ENABLE_CACHE = True
CACHE_SIZE   = 1000

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================================================
# MODULE-LEVEL SINGLETONS (loaded once on import)
# ==================================================
import torch as _torch
_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

logger.info(f"Loading {BGE_MODEL} embedder on {_DEVICE}…")
_SHARED_EMBEDDER = SentenceTransformer(BGE_MODEL, device=_DEVICE)
logger.info("✓ Shared embedder ready")

_SHARED_RERANKER = None
if USE_CROSS_ENCODER:
    try:
        logger.info(f"Loading {RERANKER_MODEL} reranker on {_DEVICE}…")
        _SHARED_RERANKER = CrossEncoder(RERANKER_MODEL, device=_DEVICE)
        logger.info(f"✓ Shared reranker ready")
    except Exception as _e:
        logger.warning(f"✗ Reranker load failed: {_e}")


@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    id: str
    content: str
    score: float
    dense_score:  Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QueryCache:
    """LRU cache for query results — Thread-safe"""

    def __init__(self, max_size: int = 1000):
        self.cache        = {}
        self.access_order = []
        self.max_size     = max_size
        self._lock        = threading.Lock()

    def _hash_query(self, query: str, filters: Optional[Dict] = None) -> str:
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        return hashlib.md5(f"{query}:{filter_str}".encode()).hexdigest()

    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[List[SearchResult]]:
        key = self._hash_query(query, filters)
        with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        return None

    def put(self, query: str, results: List[SearchResult], filters: Optional[Dict] = None):
        key = self._hash_query(query, filters)
        with self._lock:
            if len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            self.cache[key] = results
            self.access_order.append(key)

    def clear(self):
        with self._lock:
            self.cache.clear()
            self.access_order.clear()


# class BM25Encoder:
#     """BM25 query encoder — vocabulary-free hash-based fallback"""

#     def __init__(self, vocabulary: Optional[Dict[str, int]] = None):
#         self.vocabulary = vocabulary or {}

#     def encode_query(self, query: str) -> SparseVector:
#         if USE_NLTK:
#             tokens = [t.lower() for t in word_tokenize(query) if t.isalnum()]
#         else:
#             tokens = [t.lower() for t in query.split() if t.isalnum()]

#         # Vocabulary-based (legacy)
#         if self.vocabulary:
#             token_counts = {}
#             for token in tokens:
#                 if token in self.vocabulary:
#                     token_counts[token] = token_counts.get(token, 0) + 1

#             indices, values = [], []
#             for token, count in token_counts.items():
#                 indices.append(self.vocabulary[token])
#                 values.append(count / len(tokens) if tokens else 0.0)
#             return SparseVector(indices=indices, values=values)

#         # Hash-based fallback (no vocabulary needed)
#         word_freq = {}
#         for token in tokens:
#             word_freq[token] = word_freq.get(token, 0) + 1

#         indices, values = [], []
#         for word, freq in word_freq.items():
#             indices.append(hash(word) % 100000)
#             values.append(float(freq))

#         return SparseVector(indices=indices, values=values)

# pip install rank-bm25
# In Evaluate_Retrieval_With_Reranker.py — replace BM25Encoder.encode_query



class BM25Encoder:
    def __init__(self):
        self._corpus_tokens = []   # populated at index time
        self._bm25 = None

    def fit(self, texts: List[str]):
        self._corpus_tokens = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def encode_query(self, query: str) -> SparseVector:
        tokens = [t.lower() for t in query.split() if t.isalnum()]
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        indices, values = [], []
        for word, freq in token_counts.items():
            indices.append(hash(word) % 100000)
            values.append(float(freq))
        return SparseVector(indices=indices, values=values)

class QueryExpander:
    """Expand queries with synonyms and related terms"""

    def __init__(self, expansions: Dict[str, List[str]]):
        self.expansions = expansions

    def expand(self, query: str) -> str:
        if not self.expansions:
            return query
        query_lower    = query.lower()
        expanded_terms = [query]
        for key, synonyms in self.expansions.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)
        return " ".join(expanded_terms)


class MetadataFilterBuilder:
    """Build Qdrant filters from user criteria"""

    @staticmethod
    def build_filter(
        file_types:     Optional[List[str]] = None,
        filenames:      Optional[List[str]] = None,
        folders:        Optional[List[str]] = None,
        min_word_count: Optional[int]       = None,
        max_word_count: Optional[int]       = None,
        section_titles: Optional[List[str]] = None,
        has_tables:     Optional[bool]      = None,
    ) -> Optional[Filter]:
        conditions = []

        if file_types:
            conditions.append(FieldCondition(key="file_type",     match=MatchAny(any=file_types)))
        if filenames:
            conditions.append(FieldCondition(key="filename",      match=MatchAny(any=filenames)))
        if folders:
            conditions.append(FieldCondition(key="folder",        match=MatchAny(any=folders)))
        if min_word_count is not None or max_word_count is not None:
            range_filter = {}
            if min_word_count is not None:
                range_filter["gte"] = min_word_count
            if max_word_count is not None:
                range_filter["lte"] = max_word_count
            conditions.append(FieldCondition(key="word_count",    range=Range(**range_filter)))
        if section_titles:
            conditions.append(FieldCondition(key="section_title", match=MatchAny(any=section_titles)))
        if has_tables is not None:
            conditions.append(FieldCondition(key="has_tables",    match=MatchValue(value=has_tables)))

        return Filter(must=conditions) if conditions else None


class OllamaBGEM3:
    """Ollama BGE-M3 embedder — Thread-safe (fallback only)"""

    def __init__(self, base_url: str, model_name: str):
        self.base_url   = base_url
        self.model_name = model_name
        self._lock      = threading.Lock()
        self.available  = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return self.model_name in models
            return False
        except Exception:
            return False

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(t) for t in texts]

    def _get_embedding(self, text: str) -> List[float]:
        url     = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}
        try:
            with self._lock:
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
        return [0.0] * 1024

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        query_emb = self._get_embedding(query)
        scores = []
        for doc in documents:
            doc_emb = self._get_embedding(doc)
            sim = cosine_similarity(
                np.array(query_emb).reshape(1, -1),
                np.array(doc_emb).reshape(1, -1)
            )[0][0]
            scores.append(float(sim))
        return scores


class HybridRetriever:
    """
    Thread-safe hybrid retrieval system.
    Embeddings: SentenceTransformer BAAI/bge-m3 on CUDA.
    Reranker:   CrossEncoder BAAI/bge-reranker-large on CUDA (optional).

    Pass an existing `embedder` instance to reuse a pre-loaded model.
    """

    def __init__(
        self,
        qdrant_url:      str,
        collection_name: str,
        use_ollama:      bool = True,
        use_reranker:    bool = True,
        embedder:        Optional[SentenceTransformer] = None,
    ):
        self.client          = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self._qdrant_lock    = threading.Lock()

        # ── Embedder — reuse module-level singleton ───────────────────────────
        if embedder is not None:
            self.embedder = embedder
            logger.info("✓ Embedder reused from caller")
        else:
            self.embedder = _SHARED_EMBEDDER
            logger.info("✓ Embedder reused from module singleton")
        self.embedding_dim = 1024
        # ──────────────────────────────────────────────────────────────────────

        # ── Reranker — reuse module-level singleton ───────────────────────────
        self.use_reranker = use_reranker
        self.reranker     = None
        if use_reranker:
            if USE_OLLAMA_RERANKER:
                ollama = OllamaBGEM3(OLLAMA_GEN_URL, "bge-m3:latest")
                if ollama.available:
                    self.reranker = ollama
                    logger.info("✓ Reranker: Ollama BGE-M3")
            if self.reranker is None and _SHARED_RERANKER is not None:
                self.reranker = _SHARED_RERANKER
                logger.info(f"✓ Reranker reused from module singleton")
        # ──────────────────────────────────────────────────────────────────────

        self.bm25_encoder   = BM25Encoder()
        self.query_expander = QueryExpander(EXPANSION_SYNONYMS)
        self.filter_builder = MetadataFilterBuilder()
        self.cache          = QueryCache(CACHE_SIZE) if ENABLE_CACHE else None

        logger.info(f"✓ HybridRetriever ready (thread-safe) — collection: {collection_name}")

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch-encode texts on CUDA."""
        vectors = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return vectors.tolist()

    def _dense_search(
        self,
        query_vector: List[float],
        top_k:        int,
        filter_:      Optional[Filter] = None,
    ) -> List[SearchResult]:
        try:
            with self._qdrant_lock:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=filter_,
                ).points
            return [
                SearchResult(
                    id=p.id,
                    content=p.payload.get("content", ""),
                    score=p.score,
                    dense_score=p.score,
                    metadata=p.payload,
                )
                for p in results
            ]
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []

    def _sparse_search(
        self,
        query_sparse: SparseVector,
        top_k:        int,
        filter_:      Optional[Filter] = None,
    ) -> List[SearchResult]:
        if not query_sparse.indices:
            return []
        try:
            with self._qdrant_lock:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_sparse,
                    using="bm25",
                    limit=top_k,
                    query_filter=filter_,
                ).points
            return [
                SearchResult(
                    id=p.id,
                    content=p.payload.get("content", ""),
                    score=p.score,
                    sparse_score=p.score,
                    metadata=p.payload,
                )
                for p in results
            ]
        except Exception as e:
            # Silently skip — collection may not have BM25 vectors (e.g. temp collections)
            err_str = str(e)
            if "bm25" not in err_str.lower() and "vector name" not in err_str.lower():
                logger.error(f"Sparse search error: {e}")
            return []

    def _hybrid_fusion(
        self,
        dense_results:  List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight:   float = DENSE_WEIGHT,
        sparse_weight:  float = SPARSE_WEIGHT,
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion"""
        dense_map  = {r.id: (rank + 1, r) for rank, r in enumerate(dense_results)}
        sparse_map = {r.id: (rank + 1, r) for rank, r in enumerate(sparse_results)}

        combined = {}
        for doc_id in set(dense_map) | set(sparse_map):
            rrf_score = 0.0
            result    = None
            if doc_id in dense_map:
                rank, res  = dense_map[doc_id]
                rrf_score += dense_weight  / (60 + rank)
                result     = res
            if doc_id in sparse_map:
                rank, res  = sparse_map[doc_id]
                rrf_score += sparse_weight / (60 + rank)
                if result is None:
                    result = res
            combined[doc_id] = (rrf_score, result)

        fused = []
        for score, result in sorted(combined.values(), key=lambda x: x[0], reverse=True):
            # result.score = score
            if getattr(result, "rerank_score", None) is not None:
                result.score = float(result.rerank_score)
            elif getattr(result, "dense_score", None) is not None:
                result.score = float(result.dense_score)
            else:
                result.score = float(score)  # fallback RRF
            fused.append(result)
        return fused

    def _rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Rerank results using CrossEncoder or Ollama."""
        if not self.reranker or not results:
            return results[:top_k]

        try:
            documents = [r.content for r in results]

            if isinstance(self.reranker, OllamaBGEM3):
                scores = self.reranker.rerank(query, documents)
            else:
                # CrossEncoder
                pairs  = [[query, doc] for doc in documents]
                scores = self.reranker.predict(pairs, show_progress_bar=False).tolist()

            for result, score in zip(results, scores):
                result.rerank_score = float(score)

            results.sort(key=lambda x: x.rerank_score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return results[:top_k]

    def search(
        self,
        query:         str,
        top_k:         int            = FINAL_TOP_K,
        filters:       Optional[Dict] = None,
        use_expansion: bool           = False,
        use_reranking: bool           = True,
    ) -> List[SearchResult]:
        """Hybrid search with optional reranking — Thread-safe."""

        # Cache check
        if self.cache:
            cached = self.cache.get(query, filters)
            if cached:
                logger.info("✓ Cache hit")
                return cached[:top_k]

        start = time.time()

        # Query expansion
        expanded_query = (
            self.query_expander.expand(query)
            if use_expansion and ENABLE_QUERY_EXPANSION
            else query
        )

        # Metadata filter
        filter_ = self.filter_builder.build_filter(**filters) if filters else None

        # Embed query
        query_vector = self.encode_texts([expanded_query])[0]
        query_sparse = self.bm25_encoder.encode_query(expanded_query)

        # Dense + sparse search
        dense_results  = self._dense_search(query_vector, DENSE_TOP_K, filter_)
        sparse_results = self._sparse_search(query_sparse, SPARSE_TOP_K, filter_)

        logger.info(f"Dense: {len(dense_results)} | Sparse: {len(sparse_results)}")

        # Fuse
        fused = self._hybrid_fusion(dense_results, sparse_results)[:HYBRID_TOP_K]

        # Rerank
        if use_reranking and self.reranker:
            final = self._rerank(query, fused, top_k)
        else:
            final = fused[:top_k]

        logger.info(f"Search done in {(time.time()-start)*1000:.1f}ms — {len(final)} results")

        if self.cache:
            self.cache.put(query, final, filters)

        return final

    def search_with_metadata(
        self,
        query:          str,
        top_k:          int                 = 10,
        file_types:     Optional[List[str]] = None,
        filenames:      Optional[List[str]] = None,
        folders:        Optional[List[str]] = None,
        min_word_count: Optional[int]       = None,
        section_titles: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        filters = {
            "file_types":     file_types,
            "filenames":      filenames,
            "folders":        folders,
            "min_word_count": min_word_count,
            "section_titles": section_titles,
        }
        return self.search(query, top_k=top_k, filters=filters)


def main():
    logger.info("="*80)
    logger.info("HYBRID RETRIEVAL DEMO")
    logger.info("="*80)

    retriever = HybridRetriever(
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION,
    )

    query   = "What are the key features of machine learning?"
    results = retriever.search(query, top_k=5)

    logger.info(f"\nTop {len(results)} results:")
    for i, r in enumerate(results, 1):
        logger.info(f"{i}. score={r.score:.4f}  file={r.metadata.get('filename','N/A')}")
        logger.info(f"   rerank={r.rerank_score:.4f}  {r.content[:150]}…")


if __name__ == "__main__":
    main()
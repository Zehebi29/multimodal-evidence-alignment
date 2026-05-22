"""
Vector index backends for Evidence Store.

Provides pluggable storage backends:
- "json" (default): In-memory list + JSON serialization. Zero dependencies.
- "faiss" (optional): FAISS IVF/HNSW index. Requires `pip install faiss-cpu`.

Usage:
    store = EvidenceStore(api_key="...", backend="json")     # default
    store = EvidenceStore(api_key="...", backend="faiss")    # need faiss-cpu
"""

from __future__ import annotations
import json
import os
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CaseEmbedding:
    """One case in the index."""
    case_id: str
    ground_truth: str
    ai_predictions: list
    features: dict
    visual_vector: Optional[np.ndarray] = None
    text_vector: Optional[np.ndarray] = None
    fused_vector: Optional[np.ndarray] = None
    reasoning: str = ""
    supporting_evidence: list = field(default_factory=list)
    opposing_evidence: list = field(default_factory=list)
    differential: list = field(default_factory=list)
    uncertainty_sources: list = field(default_factory=list)


class IndexBackend(ABC):
    """Abstract interface for vector index backends."""

    def __init__(self, dim: int = 5120):
        self.dim = dim

    @abstractmethod
    def add(self, entry: CaseEmbedding):
        """Add a case to the index."""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[float, int]]:
        """Search top-K. Returns [(similarity, index_position), ...]."""
        ...

    @abstractmethod
    def get(self, idx: int) -> CaseEmbedding:
        """Get case by index position."""
        ...

    @abstractmethod
    def save(self, path: str):
        """Persist index to disk."""
        ...

    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


# ═══════════════════════════════════════════════════════════
# JSON Backend (default)
# ═══════════════════════════════════════════════════════════

class InMemoryBackend(IndexBackend):
    """In-memory list + JSON serialization. Default, zero extra deps.

    Search is brute-force O(n) cosine similarity. Fine for <10K cases.
    """

    def __init__(self, dim: int = 5120):
        super().__init__(dim)
        self._entries: list[CaseEmbedding] = []

    def add(self, entry: CaseEmbedding):
        self._entries.append(entry)

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, int]]:
        scored = []
        for i, entry in enumerate(self._entries):
            if entry.fused_vector is None:
                continue
            sim = float(np.dot(query, entry.fused_vector) /
                        (np.linalg.norm(query) * np.linalg.norm(entry.fused_vector) + 1e-10))
            scored.append((sim, i))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def get(self, idx: int) -> CaseEmbedding:
        return self._entries[idx]

    def save(self, path: str):
        data = []
        for e in self._entries:
            d = {
                "case_id": e.case_id, "ground_truth": e.ground_truth,
                "ai_predictions": e.ai_predictions, "features": e.features,
                "reasoning": e.reasoning,
                "supporting_evidence": e.supporting_evidence,
                "opposing_evidence": e.opposing_evidence,
                "differential": e.differential,
                "uncertainty_sources": e.uncertainty_sources,
            }
            if e.fused_vector is not None:
                d["fused_vector"] = e.fused_vector.tolist()
            if e.visual_vector is not None:
                d["visual_vector"] = e.visual_vector.tolist()
            if e.text_vector is not None:
                d["text_vector"] = e.text_vector.tolist()
            data.append(d)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._entries = []
        for d in data:
            e = CaseEmbedding(
                case_id=d["case_id"], ground_truth=d["ground_truth"],
                ai_predictions=d["ai_predictions"], features=d["features"],
                reasoning=d.get("reasoning", ""),
                supporting_evidence=d.get("supporting_evidence", []),
                opposing_evidence=d.get("opposing_evidence", []),
                differential=d.get("differential", []),
                uncertainty_sources=d.get("uncertainty_sources", []),
            )
            if "fused_vector" in d:
                e.fused_vector = np.array(d["fused_vector"], dtype=np.float32)
            if "visual_vector" in d:
                e.visual_vector = np.array(d["visual_vector"], dtype=np.float32)
            if "text_vector" in d:
                e.text_vector = np.array(d["text_vector"], dtype=np.float32)
            self._entries.append(e)

    def __len__(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════
# FAISS Backend (optional)
# ═══════════════════════════════════════════════════════════

class FAISSBackend(IndexBackend):
    """FAISS IVF-based index. Requires `pip install faiss-cpu`.

    Much faster for >10K cases. Uses IVF4096 + Flat quantizer.
    """

    def __init__(self, dim: int = 5120, nlist: int = 128):
        super().__init__(dim)
        self._faiss = None
        self._index = None
        self._entries: list[CaseEmbedding] = []
        self._nlist = nlist  # IVF clusters (auto-capped to sqrt(N))

    def _ensure_faiss(self):
        if self._faiss is not None:
            return
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS backend requires: pip install faiss-cpu\n"
                "Or use backend='json' (default) for zero-dependency operation."
            )

    def _build_index(self):
        if not self._entries:
            return
        self._ensure_faiss()
        vectors = np.stack([e.fused_vector for e in self._entries
                            if e.fused_vector is not None]).astype(np.float32)
        n = len(vectors)
        nlist = min(self._nlist, int(np.sqrt(n)) + 1)

        quantizer = self._faiss.IndexFlatIP(self.dim)  # inner product (= cosine on normalized)
        self._index = self._faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                                self._faiss.METRIC_INNER_PRODUCT)
        self._index.train(vectors)
        self._index.add(vectors)

    def add(self, entry: CaseEmbedding):
        self._entries.append(entry)
        self._index = None  # invalidate, rebuild on next search

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, int]]:
        self._ensure_faiss()
        if self._index is None:
            self._build_index()
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query.astype(np.float32).reshape(1, -1)
        # Normalize for inner product = cosine
        self._faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._entries):
                results.append((float(score), int(idx)))
        return results

    def get(self, idx: int) -> CaseEmbedding:
        return self._entries[idx]

    def save(self, path: str):
        """Save FAISS index + metadata to disk."""
        self._ensure_faiss()
        if self._index is None:
            self._build_index()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Save FAISS index
        faiss_path = path.replace(".json", ".faiss")
        if self._index is not None:
            self._faiss.write_index(self._index, faiss_path)

        # Save metadata (entries without vectors)
        data = []
        for e in self._entries:
            d = {
                "case_id": e.case_id, "ground_truth": e.ground_truth,
                "ai_predictions": e.ai_predictions, "features": e.features,
                "reasoning": e.reasoning,
                "supporting_evidence": e.supporting_evidence,
                "opposing_evidence": e.opposing_evidence,
                "differential": e.differential,
                "uncertainty_sources": e.uncertainty_sources,
            }
            data.append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        """Load FAISS index + metadata from disk."""
        self._ensure_faiss()

        # Load FAISS index
        faiss_path = path.replace(".json", ".faiss")
        if os.path.exists(faiss_path):
            self._index = self._faiss.read_index(faiss_path)
        else:
            self._index = None

        # Load metadata
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._entries = []
        for d in data:
            e = CaseEmbedding(
                case_id=d["case_id"], ground_truth=d["ground_truth"],
                ai_predictions=d["ai_predictions"], features=d["features"],
                reasoning=d.get("reasoning", ""),
                supporting_evidence=d.get("supporting_evidence", []),
                opposing_evidence=d.get("opposing_evidence", []),
                differential=d.get("differential", []),
                uncertainty_sources=d.get("uncertainty_sources", []),
            )
            self._entries.append(e)

        # Reconstruct fused vectors from FAISS index if available
        if self._index is not None:
            for i, e in enumerate(self._entries):
                if i < self._index.ntotal:
                    vec = np.zeros(self.dim, dtype=np.float32)
                    self._index.reconstruct(i, vec)
                    e.fused_vector = vec

    def __len__(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════

def create_backend(name: str = "json", dim: int = 5120, **kwargs) -> IndexBackend:
    """Create an index backend by name.

    Args:
        name: "json" (default) or "faiss"
        dim: Vector dimension
        **kwargs: Backend-specific options (e.g., nlist for FAISS)

    Returns:
        IndexBackend instance
    """
    if name == "json":
        return InMemoryBackend(dim=dim)
    elif name == "faiss":
        nlist = kwargs.get("nlist", 128)
        return FAISSBackend(dim=dim, nlist=nlist)
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'json' or 'faiss'.")

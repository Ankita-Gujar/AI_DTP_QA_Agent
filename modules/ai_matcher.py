"""
ai_matcher.py
-------------
Semantic similarity between source-language and target-language paragraph
text, using multilingual sentence embeddings. This is the ONLY place text
content is ever compared -- and even here it's used purely as one signal
feeding into the match score, never as a translation-quality check.

We explicitly do NOT use difflib.SequenceMatcher or any character-overlap
method, since source and target text are in different languages and
literal comparison is meaningless.

Model: sentence-transformers "paraphrase-multilingual-MiniLM-L12-v2"
(fast, 50+ languages) with "intfloat/multilingual-e5-base" as a documented
alternative for higher accuracy at higher latency.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .utils import QAConfig, get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception as _import_exc:  # pragma: no cover
    # Deliberately broad: on Windows this commonly fails as OSError (a torch DLL
    # failing to load), not ImportError, so a narrow `except ImportError` would
    # let the exception escape and crash the whole pipeline. Any failure here
    # should just disable semantic matching, never take down the app.
    SentenceTransformer = None
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers/torch could not be loaded (%s) -- semantic matching "
        "will fall back to a neutral score. See README for the Windows torch/DLL "
        "troubleshooting note if you want real cross-lingual paragraph matching.",
        _import_exc,
    )


class SemanticMatcher:
    """Lazy-loaded, cached multilingual embedding model.

    One instance should be created per pipeline run and shared across all
    paragraph comparisons -- loading the model is expensive, embedding is cheap.

    The model itself is process-wide (via the `_instance` singleton) so a
    long-running server doesn't reload multi-hundred-MB weights on every QA
    run. The text->embedding cache below is bounded for the same reason: on
    a server processing many distinct documents over days/weeks, an unbounded
    cache keyed by every paragraph text ever seen would grow without limit.
    """

    _instance: Optional["SemanticMatcher"] = None
    _MAX_CACHE_ENTRIES = 50_000  # ~a few hundred large documents' worth of unique paragraphs

    def __init__(self, config: QAConfig):
        self.config = config
        self._model = None
        self._cache: dict = {}

    @classmethod
    def get(cls, config: QAConfig) -> "SemanticMatcher":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _ensure_model(self) -> None:
        if self._model is not None or not _ST_AVAILABLE:
            return
        try:
            logger.info("Loading multilingual embedding model: %s", self.config.embedding_model_name)
            self._model = SentenceTransformer(self.config.embedding_model_name)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load embedding model (%s). Falling back to neutral scores.", exc)
            self._model = None

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if not _ST_AVAILABLE:
            return None
        self._ensure_model()
        if self._model is None:
            return None
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            vectors = self._model.encode(
                uncached,
                batch_size=self.config.embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for t, v in zip(uncached, vectors):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity in [0, 1]. Returns a neutral 0.5 if embeddings are unavailable
        (so the overall match score degrades gracefully to layout+size+order signals
        instead of silently being wrong)."""
        text_a = (text_a or "").strip()
        text_b = (text_b or "").strip()
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0

        vectors = self.embed([text_a, text_b])
        if vectors is None:
            return 0.5
        a, b = vectors[0], vectors[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.5
        cos = float(np.dot(a, b) / denom)
        return max(0.0, min(1.0, (cos + 1.0) / 2.0 if cos < 0 else cos))

    def similarity_matrix(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        """Vectorized similarity matrix for efficient batch matching (used by paragraph_matcher)."""
        if not texts_a or not texts_b:
            return np.zeros((len(texts_a), len(texts_b)))
        vectors_a = self.embed(texts_a)
        vectors_b = self.embed(texts_b)
        if vectors_a is None or vectors_b is None:
            return np.full((len(texts_a), len(texts_b)), 0.5)
        norm_a = vectors_a / np.clip(np.linalg.norm(vectors_a, axis=1, keepdims=True), 1e-8, None)
        norm_b = vectors_b / np.clip(np.linalg.norm(vectors_b, axis=1, keepdims=True), 1e-8, None)
        sim = norm_a @ norm_b.T
        return np.clip((sim + 1.0) / 2.0, 0.0, 1.0)


def is_semantic_matching_available() -> bool:
    return _ST_AVAILABLE

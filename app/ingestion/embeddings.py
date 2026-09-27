"""Embedding providers for retrieval.

The default provider is a deterministic, dependency-free feature-hashing
embedder. It needs no model download, runs instantly on CPU, and is good enough
for lexical/semantic-ish recall on a personal knowledge base. It is labelled
``local-hashing-v1`` in the API so nobody mistakes it for a neural model.

``sentence-transformers`` is supported as an opt-in upgrade for users who want
semantic recall and are willing to install and download a small model.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

from app.core.logging import get_logger

logger = get_logger("msn_ai.ingestion.embeddings")

#: Word-ish tokens; keeps identifiers like ``create_chat_completion`` intact.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
#: Extra credit for the bigrams of adjacent tokens (cheap word-order signal).
_BIGRAMS_ENABLED = True

#: Function words carry no retrieval signal but would otherwise dominate short
#: queries, so they are down-weighted rather than dropped (they still help
#: discriminate two otherwise identical sentences).
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those is are was were be been being
    am do does did done have has had having will would shall should can could may might must
    i me my we our you your he she it they them their what which who whom whose when where
    why how all any both each few more most other some such no nor not only own same so too
    very just about into over after before between out up down off again further once here
    there also as of to in on at by for from with without
    """.split()
)
#: Weight applied to stopwords versus content words.
_STOPWORD_WEIGHT = 0.25


_SIBILANT = frozenset("sxz")


def stem(token: str) -> str:
    """Very small suffix stripper so plurals and tenses collide.

    A real stemmer (Porter/Snowball) is overkill here and would need a
    dependency; this covers the cases that actually matter for retrieval:
    ``files -> file``, ``limits -> limit``, ``forgetting -> forget``,
    ``configured -> configure``, ``directories -> directory``.

    ``-es`` is only removed after a sibilant so ``files`` stems to ``file``
    (not ``fil``) while ``boxes`` still stems to ``box``.
    """
    if len(token) <= 3 or token in _STOPWORDS:
        return token
    for suffix, minimum in (
        ("ies", 5),
        ("ing", 5),
        ("ed", 4),
        ("es", 4),
        ("s", 4),
    ):
        if not (token.endswith(suffix) and len(token) >= minimum):
            continue
        base = token[: -len(suffix)]
        if suffix == "ies":
            return base + "y"
        if suffix == "s" and base.endswith("s"):
            return token  # 'address' -> 'addres' would be wrong
        if suffix == "es" and base[-1:] not in _SIBILANT:
            continue  # 'files' -> strip only the 's' below
        # Undo a doubled consonant: 'running' -> 'runn' -> 'run'.
        if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            base = base[:-1]
        return base
    return token


def tokenize(text: str) -> list[str]:
    """Lowercased, stemmed tokens plus camelCase/snake_case sub-tokens."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        raw = match.group(0)
        token = raw.lower()
        # Split snake_case and camelCase so "createChatCompletion" matches
        # "create chat completion" too.
        parts = [p for p in re.split(r"_|__", raw) if p]
        pieces = [token] + [p.lower() for p in parts] if len(parts) > 1 else [token]
        if any(ch.isupper() for ch in raw[1:]):
            pieces += re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", raw)
        seen: set[str] = set()
        for piece in pieces:
            low = piece.lower()
            if low and low not in seen:
                seen.add(low)
                tokens.append(low)
        stemmed = stem(token)
        if stemmed != token and stemmed not in seen:
            tokens.append(stemmed)
    return tokens


class EmbeddingProvider(ABC):
    """Turn text into a fixed-length unit vector."""

    name: str = "base"
    dim: int = 384

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single string."""

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "available": True,
            "reason": "",
            "dim": self.dim,
        }


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-features embedding (no model, no download).

    Uses feature hashing into ``dim`` buckets with sublinear term frequency and
    stopword down-weighting, then L2-normalises. Buckets are **unsigned**: a
    personal knowledge base holds thousands of short chunks, not millions of
    documents, so the variance that signed hashing removes costs more than its
    bias does -- signed hashing was measurably ranking relevant pairs below
    unrelated ones.

    Output is deterministic across processes and machines, which matters because
    vectors are cached on disk.
    """

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = max(int(dim), 32)

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector

        weighted: dict[str, float] = {}
        for token in tokens:
            weight = _STOPWORD_WEIGHT if token in _STOPWORDS else 1.0
            weighted[token] = weighted.get(token, 0.0) + weight
        if _BIGRAMS_ENABLED:
            for left, right in zip(tokens, tokens[1:]):
                bigram = f"{left}\x1f{right}"
                # Bigrams of two stopwords carry no signal.
                if left in _STOPWORDS and right in _STOPWORDS:
                    continue
                weighted[bigram] = weighted.get(bigram, 0.0) + 0.5

        for feature, weight in weighted.items():
            # Sublinear weighting damps repeated words without losing order.
            vector[self._bucket(feature)] += math.copysign(
                1.0 + math.log(1.0 + weight), weight
            ) * abs(weight) ** 0.5

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": "local-hashing-v1",
            "available": True,
            "reason": "",
            "dim": self.dim,
            "requires_download": False,
            "note": "Deterministic local feature hashing; no neural model is used.",
        }


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Opt-in semantic embeddings from a local sentence-transformers model."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._dim = 0
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name, device=device)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:
            logger.info("sentence-transformers unavailable: %s", exc)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim or 384

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError(
                "sentence-transformers is not installed or the model could not be "
                "loaded. Install it with: pip install sentence-transformers"
            )
        vector = self._model.encode(text or "", convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in vector]

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model_name,
            "dim": self.dim,
            "available": self.available,
            "reason": "" if self.available else (
                "sentence-transformers is not installed or the model is not "
                "available locally."
            ),
            "requires_download": True,
        }


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for two already-L2-normalised or raw vectors."""
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(size))
    na = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(size)))
    nb = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(size)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_embedding_provider(settings) -> EmbeddingProvider:
    """Instantiate the embedding provider named in settings.

    Falls back to hashing (with a warning) when a requested provider is missing,
    because a working knowledge base beats a broken one.
    """
    provider_name = (getattr(settings, "embedding_provider", "hashing") or "hashing").lower()
    dim = int(getattr(settings, "embedding_dim", 384) or 384)

    if provider_name in {"hashing", "local-hashing", "default"}:
        return HashingEmbeddingProvider(dim=dim)
    if provider_name in {"sentence-transformers", "st", "minilm"}:
        provider = SentenceTransformerEmbeddingProvider(
            model_name=getattr(settings, "embedding_model", "all-MiniLM-L6-v2")
        )
        if provider.available:
            return provider
        logger.warning(
            "Embedding provider '%s' is unavailable; using local hashing instead.",
            provider_name,
        )
        return HashingEmbeddingProvider(dim=dim)
    return HashingEmbeddingProvider(dim=dim)

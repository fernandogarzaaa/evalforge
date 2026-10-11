"""Text embedders.

The default ``HashingEmbedder`` is stateless: the same text maps to the same vector on
every nightly run, on every machine, with no model download. That stability matters
more here than semantic depth, because embeddings are persisted (test-case dedup,
category centroids) and compared across runs. An OpenAI-compatible embeddings
endpoint can be swapped in via ``OpenAIEmbedder`` when better recall is worth the cost.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer

PLACEHOLDER = re.compile(r"\[(EMAIL|PHONE|CARD|IP|API_KEY)\]")
# Greetings and sign-offs carry no task meaning but would keep paraphrases apart.
CHITCHAT = frozenset(
    {"hi", "hey", "hello", "thanks", "thank", "thx", "asap", "quick", "question", "team", "please", "pls", "cheers"}
)
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | CHITCHAT)


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def normalise_text(text: str) -> str:
    """Lower-case and drop redaction placeholders so PII does not separate otherwise-equal prompts."""
    return PLACEHOLDER.sub(" ", text).lower()


class HashingEmbedder:
    def __init__(self, n_features: int = 2048) -> None:
        self.name = f"hashing-{n_features}"
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
            preprocessor=normalise_text,
            token_pattern=r"(?u)\b\w\w+\b",  # noqa: S106 - regex, not a secret
            stop_words=STOP_WORDS,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.vectorizer.n_features), dtype=np.float32)
        return np.asarray(self.vectorizer.transform(list(texts)).toarray(), dtype=np.float32)


class OpenAIEmbedder:
    """Any OpenAI-compatible ``/embeddings`` endpoint (OPENAI_BASE_URL, OPENAI_API_KEY, EMBED_MODEL)."""

    def __init__(self, base_url: str, api_key: str, model: str, client: httpx.Client | None = None) -> None:
        self.name = f"openai:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client or httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key}"})

    @classmethod
    def from_env(cls) -> OpenAIEmbedder:
        return cls(
            os.environ["OPENAI_BASE_URL"],
            os.environ.get("OPENAI_API_KEY", ""),
            os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            resp = self.client.post(
                f"{self.base_url}/embeddings", json={"model": self.model, "input": list(texts[i : i + 128])}
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    """Local ONNX sentence embeddings via ``fastembed`` (``pip install evalforge[semantic]``).

    Free and offline after a one-time ~70 MB model download; deterministic for a fixed model.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # optional dependency

        self.name = f"fastembed:{model}"
        cache = cache_dir or os.environ.get("EVALFORGE_CACHE", os.path.expanduser("~/.cache/evalforge"))
        self.model = TextEmbedding(model, cache_dir=cache)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        cleaned = [PLACEHOLDER.sub(" ", t) for t in texts]
        arr = np.asarray(list(self.model.embed(cleaned)), dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


class CachedEmbedder:
    """Memoises another embedder by exact text; optionally persisted to an ``.npz`` file.

    Production prompts repeat a lot (templates, retries), so this saves real compute.
    """

    def __init__(self, inner: Embedder, path: str | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.path = path
        self.cache: dict[str, np.ndarray] = {}
        if path and os.path.exists(path):
            data = np.load(path, allow_pickle=False)
            self.cache = dict(zip(data["texts"].tolist(), data["vectors"]))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            for t, v in zip(missing, self.inner.embed(missing)):
                self.cache[t] = v
            if self.path:
                keys = list(self.cache)
                np.savez(self.path, texts=np.asarray(keys), vectors=np.stack([self.cache[k] for k in keys]))
        if not texts:
            return self.inner.embed(texts)
        return np.stack([self.cache[t] for t in texts])


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of ``a`` and rows of ``b`` (rows need not be normalised)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.asarray(an @ bn.T)


def get_embedder(kind: str | None = None) -> Embedder:
    kind = kind or os.environ.get("EVALFORGE_EMBEDDER", "hashing")
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    if kind == "openai":
        return OpenAIEmbedder.from_env()
    raise ValueError(f"unknown embedder: {kind}")

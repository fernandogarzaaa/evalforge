"""Targeted tests for evalforge.embed that pin behaviour mutation testing found unguarded.

The optional ``FastEmbedEmbedder`` path is exercised with a fake ``fastembed`` module, so no
model download or network access is needed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import httpx
import numpy as np
import pytest

from evalforge.embed import (
    CHITCHAT,
    PLACEHOLDER,
    STOP_WORDS,
    CachedEmbedder,
    FastEmbedEmbedder,
    HashingEmbedder,
    OpenAIEmbedder,
    cosine_matrix,
    get_embedder,
    normalise_text,
)


class _FakeTextEmbedding:
    instances: list[_FakeTextEmbedding] = []

    def __init__(self, model: str, cache_dir: str | None = None) -> None:
        self.model_name = model
        self.cache_dir = cache_dir
        self.seen: list[list[str]] = []
        _FakeTextEmbedding.instances.append(self)

    def embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        self.seen.append(list(texts))
        for t in texts:
            # a deterministic, deliberately un-normalised vector
            yield np.array([float(len(t)), 0.0, 0.0] + [0.0] * 381, dtype=np.float32) + np.eye(384)[1] * 3


@pytest.fixture()
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTextEmbedding]:
    _FakeTextEmbedding.instances = []
    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    return _FakeTextEmbedding


def test_fastembed_default_model_and_cache_dir(fake_fastembed, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("EVALFORGE_CACHE", raising=False)
    monkeypatch.setenv("HOME", "/home/someone")
    e = FastEmbedEmbedder()
    inst = fake_fastembed.instances[-1]
    assert e.name == "fastembed:BAAI/bge-small-en-v1.5"
    assert inst.model_name == "BAAI/bge-small-en-v1.5"
    assert inst.cache_dir == "/home/someone/.cache/evalforge"
    assert e.model is inst


def test_fastembed_cache_dir_precedence(fake_fastembed, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("EVALFORGE_CACHE", "/env/cache")
    FastEmbedEmbedder("m")
    assert fake_fastembed.instances[-1].cache_dir == "/env/cache"
    FastEmbedEmbedder("m", cache_dir="/explicit")
    assert fake_fastembed.instances[-1].cache_dir == "/explicit"
    assert fake_fastembed.instances[-1].model_name == "m"


def test_fastembed_embed_strips_placeholders_and_normalises(fake_fastembed) -> None:  # type: ignore[no-untyped-def]
    e = FastEmbedEmbedder("m", cache_dir="/c")
    out = e.embed(["mail [EMAIL] now", "abcd"])
    inst = fake_fastembed.instances[-1]
    # placeholders are replaced with a space, case is preserved (unlike normalise_text)
    assert inst.seen == [["mail   now", "abcd"]]
    assert out.shape == (2, 384) and out.dtype == np.float32
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0], abs=1e-6)
    # [10,3] / 10.44 and [4,3] / 5
    assert out[1, 0] == pytest.approx(0.8, abs=1e-6) and out[1, 1] == pytest.approx(0.6, abs=1e-6)


def test_fastembed_empty_input_does_not_call_model(fake_fastembed) -> None:  # type: ignore[no-untyped-def]
    e = FastEmbedEmbedder("m", cache_dir="/c")
    out = e.embed([])
    assert out.shape == (0, 384) and out.dtype == np.float32
    assert fake_fastembed.instances[-1].seen == []


def test_fastembed_zero_vector_is_not_nan(fake_fastembed) -> None:  # type: ignore[no-untyped-def]
    e = FastEmbedEmbedder("m", cache_dir="/c")

    class Zero:
        def embed(self, texts):  # type: ignore[no-untyped-def]
            for _ in texts:
                yield np.zeros(384, dtype=np.float32)

    e.model = Zero()  # type: ignore[assignment]
    out = e.embed(["x"])
    assert not np.isnan(out).any() and float(np.abs(out).sum()) == 0.0


def test_get_embedder_fastembed_and_env(fake_fastembed, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(get_embedder("fastembed"), FastEmbedEmbedder)
    monkeypatch.setenv("EVALFORGE_EMBEDDER", "fastembed")
    assert isinstance(get_embedder(), FastEmbedEmbedder)
    monkeypatch.setenv("EVALFORGE_EMBEDDER", "hashing")
    assert isinstance(get_embedder(), HashingEmbedder)
    with pytest.raises(ValueError, match="unknown embedder: nope"):
        get_embedder("nope")


def test_placeholder_and_chitchat_constants() -> None:
    for tag in ("EMAIL", "PHONE", "CARD", "IP", "API_KEY"):
        assert PLACEHOLDER.fullmatch(f"[{tag}]")
    assert not PLACEHOLDER.search("[NAME]")
    assert {"hi", "thanks", "please", "cheers"} <= CHITCHAT
    assert sorted(STOP_WORDS) == STOP_WORDS and "the" in STOP_WORDS and "thx" in STOP_WORDS
    assert normalise_text("A[PHONE]B") == "a b"


def test_hashing_embedder_config() -> None:
    e = HashingEmbedder(n_features=64)
    assert e.name == "hashing-64"
    v = e.vectorizer
    assert v.n_features == 64 and v.ngram_range == (1, 2) and v.alternate_sign is False and v.norm == "l2"
    out = e.embed(["reset password", "billing invoice"])
    assert out.shape == (2, 64) and out.dtype == np.float32
    assert (out >= 0).all()  # alternate_sign=False
    # single-character tokens are ignored by the token pattern; stop words too
    assert float(np.abs(e.embed(["a b c the"])).sum()) == 0.0
    # bigrams contribute: word order changes the vector
    a, b = e.embed(["reset password"]), e.embed(["password reset"])
    assert not np.allclose(a, b)


def test_openai_embedder_request_shape_and_zero_norm() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [0.0, 0.0]} for i in range(len(body["input"]))]}
        )

    e = OpenAIEmbedder("http://h/v1///", "k", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert e.base_url == "http://h/v1" and e.model == "m"
    out = e.embed([f"t{i}" for i in range(128)])
    assert len(seen) == 1 and str(seen[0].url) == "http://h/v1/embeddings"
    assert out.shape == (128, 2) and not np.isnan(out).any() and out.dtype == np.float32


def test_openai_embedder_orders_by_index_and_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [5.0, 0.0]}]}
        )

    e = OpenAIEmbedder("http://h", "k", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert e.embed(["a", "b"]).tolist() == [[1.0, 0.0], [0.0, 1.0]]

    bad = OpenAIEmbedder(
        "http://h", "k", "m", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    )
    with pytest.raises(httpx.HTTPStatusError):
        bad.embed(["a"])


def test_openai_embedder_default_client_and_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    e = OpenAIEmbedder("http://h", "secret", "m")
    assert e.client.headers["Authorization"] == "Bearer secret"
    assert e.client.timeout.read == 60
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env/")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    f = OpenAIEmbedder.from_env()
    assert f.base_url == "http://env" and f.model == "text-embedding-3-small"
    assert f.client.headers["Authorization"] == "Bearer "
    monkeypatch.setenv("EMBED_MODEL", "x")
    assert OpenAIEmbedder.from_env().name == "openai:x"


def test_cached_embedder_without_path_and_missing_file(tmp_path: Path) -> None:
    class Inner:
        name = "i"
        calls = 0

        def embed(self, texts):  # type: ignore[no-untyped-def]
            Inner.calls += 1
            return np.asarray([[float(len(t))] for t in texts], dtype=np.float32).reshape(len(texts), 1)

    c = CachedEmbedder(Inner())
    assert c.path is None and c.cache == {}
    assert c.embed(["ab", "ab"]).tolist() == [[2.0], [2.0]] and Inner.calls == 1
    assert c.embed(["ab"]).tolist() == [[2.0]] and Inner.calls == 1  # fully cached: no inner call
    p = tmp_path / "missing.npz"
    d = CachedEmbedder(Inner(), str(p))
    assert d.cache == {}
    d.embed(["xyz"])
    assert p.exists()


def test_cosine_matrix_unnormalised_and_empty_right() -> None:
    m = cosine_matrix(np.array([[2.0, 0.0]]), np.array([[0.0, 5.0], [7.0, 0.0], [-1.0, 0.0]]))
    assert m[0].tolist() == pytest.approx([0.0, 1.0, -1.0])
    assert cosine_matrix(np.ones((3, 2)), np.zeros((0, 2))).shape == (3, 0)
    tiny = cosine_matrix(np.array([[1e-13, 0.0]]), np.array([[1.0, 0.0]]))
    assert tiny[0, 0] == pytest.approx(0.1, rel=1e-3)  # clamped at 1e-12, not divided by the tiny norm

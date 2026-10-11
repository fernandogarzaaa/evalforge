"""Targeted tests for evalforge.cluster (mutation-testing survivors): size guards, thresholds, merging."""

from __future__ import annotations

import numpy as np
import pytest

from evalforge.cluster import (
    EXTRA_STOP,
    assign_categories,
    cluster_embeddings,
    name_cluster,
    representative,
    top_terms,
)
from evalforge.embed import cosine_matrix


def _blobs(per: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros(64)
    a[0] = 1.0
    b = np.zeros(64)
    b[32] = 1.0
    pts = [a + rng.normal(0, 0.01, 64) for _ in range(per)] + [b + rng.normal(0, 0.01, 64) for _ in range(per)]
    return np.asarray(pts, dtype=np.float32)


def test_cluster_embeddings_size_guard_boundaries() -> None:
    # fewer rows than min_cluster_size: all noise, int dtype
    out = cluster_embeddings(_blobs(2), min_cluster_size=5)
    assert out.tolist() == [-1] * 4 and out.dtype.kind == "i"
    # fewer than 3 rows is always noise, even with a tiny min_cluster_size
    assert cluster_embeddings(np.ones((2, 8)), min_cluster_size=2).tolist() == [-1, -1]
    # exactly min_cluster_size rows per blob is enough to cluster
    labels = cluster_embeddings(_blobs(5), min_cluster_size=5)
    assert labels.dtype.kind == "i" and len(labels) == 10
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1 and labels[0] != labels[5] and -1 not in labels


def test_cluster_embeddings_default_min_size_and_low_dim_input() -> None:
    x = _blobs(6)
    labels = cluster_embeddings(x)  # default min_cluster_size=5
    assert labels[0] != labels[6] and -1 not in labels
    assert cluster_embeddings(_blobs(2)).tolist() == [-1] * 4  # 4 rows < default 5
    # two features: no SVD (dims would be 1), points used directly
    two = np.asarray([[1, 0]] * 4 + [[0, 1]] * 4, dtype=np.float32) + np.linspace(0, 0.01, 8)[:, None]
    lab = cluster_embeddings(two, min_cluster_size=3)
    assert lab[0] != lab[4] and -1 not in lab
    # deterministic for a fixed seed
    assert cluster_embeddings(x, 3, seed=1).tolist() == cluster_embeddings(x, 3, seed=1).tolist()


def test_cluster_embeddings_unnormalised_scale_does_not_matter() -> None:
    x = _blobs(6)
    x[:6] *= 100.0
    labels = cluster_embeddings(x, 3)
    assert len(set(labels[:6])) == 1 and labels[0] != labels[6]


def test_top_terms_rules() -> None:
    # min length 3, counted once per text, ties broken alphabetically, k respected
    texts = ["zip zip zip ab abc", "abc zip", "cat"]
    assert top_terms(texts, k=2) == ["abc", "zip"]
    assert top_terms(texts) == ["abc", "zip", "cat"]
    assert top_terms(texts, k=1) == ["abc"]
    # words must start with a letter; digits/underscores allowed after
    assert top_terms(["9abc x_1 a_1b"]) == ["a_1b", "abc", "x_1"]
    assert top_terms(["ab 12 a1"]) == []
    # stop words (sklearn and EXTRA_STOP) are dropped, placeholders removed, case folded
    assert top_terms(["Please HELP with Billing [EMAIL] the"]) == ["billing"]
    assert {"acme", "want", "use", "make"} <= EXTRA_STOP


def test_name_cluster_joins_three_terms() -> None:
    assert name_cluster(["alpha beta gamma delta", "alpha beta gamma", "alpha beta"]) == "alpha / beta / gamma"
    assert name_cluster([]) == "misc"


def test_representative_uses_centroid_direction() -> None:
    x = np.array([[10.0, 0.0], [1.0, 1.1], [0.0, 10.0]])
    assert representative(["a", "b", "c"], x) == "b"
    assert representative(["a", "b"], np.array([[5.0, 0.1], [0.0, 1.0]])) == "a"


def _unit(i: int, d: int = 4) -> np.ndarray:
    v = np.zeros(d, dtype=np.float32)
    v[i] = 1.0
    return v


def test_cluster_match_threshold_is_inclusive() -> None:
    x = np.asarray([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]], dtype=np.float32)
    existing = [{"id": 3, "name": "e", "centroid": _unit(0).tolist(), "size": 2}]
    sim = float(cosine_matrix(x[:1], np.asarray([existing[0]["centroid"]]))[0, 0])
    labels = np.array([0, 0])
    at = assign_categories(["p", "q"], x, labels, existing, match_threshold=sim)
    assert at.updates[0]["id"] == 3 and at.updates[0]["name"] == "e"
    above = assign_categories(["p", "q"], x, labels, existing, match_threshold=sim + 1e-4)
    assert above.updates[0]["id"] is None
    # default threshold 0.7: cos = 0.7071 matches
    assert assign_categories(["p", "q"], x, labels, existing).updates[0]["id"] == 3


def test_default_thresholds() -> None:
    x = np.asarray([[1.0, 1.0, 0, 0], [1.0, 1.0, 0, 0]], dtype=np.float32)
    existing = [{"id": 3, "name": "e", "centroid": _unit(0).tolist(), "size": 2}]
    # noise at cos 0.7071 does not join (default noise threshold 0.8) ...
    assert assign_categories(["p", "q"], x, np.array([-1, -1]), existing).category_ids == [None, None]
    # ... but cos 0.83 does
    y = np.asarray([[1.5, 1.0, 0, 0]], dtype=np.float32)
    assert assign_categories(["p"], y, np.array([-1]), existing).category_ids == [3]
    # a cluster at cos 0.69 founds a new category under the default 0.7
    z = np.asarray([[1.0, 1.05, 0, 0]] * 2, dtype=np.float32)
    assert float(cosine_matrix(z[:1], np.asarray([_unit(0)]))[0, 0]) < 0.7
    assert assign_categories(["p", "q"], z, np.array([0, 0]), existing).updates[0]["id"] is None


def test_noise_threshold_is_inclusive_and_picks_best() -> None:
    x = np.asarray([[1.0, 0.2, 0, 0]], dtype=np.float32)
    existing = [
        {"id": 1, "name": "a", "centroid": _unit(1).tolist(), "size": 1},
        {"id": 9, "name": "b", "centroid": _unit(0).tolist(), "size": 1},
    ]
    sim = float(cosine_matrix(x, np.asarray([_unit(0)]))[0, 0])
    out = assign_categories(["n"], x, np.array([-1]), existing, noise_threshold=sim)
    assert out.category_ids == [9] and out.n_noise == 1 and out.n_clusters == 0 and out.updates == []
    assert assign_categories(["n"], x, np.array([-1]), existing, noise_threshold=sim + 1e-4).category_ids == [None]


def test_merge_is_size_weighted_and_new_category_fields() -> None:
    x = np.asarray([_unit(0), _unit(0), _unit(0), _unit(1)], dtype=np.float32)
    existing = [{"id": 5, "name": "old", "centroid": [0.0, 1.0, 0.0, 0.0], "size": 3}]
    out = assign_categories(["a", "b", "c", "d"], x, np.array([0, 0, 0, -1]), existing, match_threshold=-1.0)
    u = out.updates[0]
    assert u["id"] == 5 and u["size"] == 6 and u["members"] == [0, 1, 2]
    assert u["centroid"] == pytest.approx([0.5, 0.5, 0.0, 0.0])
    # noise is compared to the merged centroid (0.5, 0.5): cos 0.707 < 0.8, so d stays unclustered
    assert out.category_ids == [None, None, None, None]
    assert existing[0]["size"] == 3 and existing[0]["centroid"] == [0.0, 1.0, 0.0, 0.0]
    fresh = assign_categories(["alpha bravo", "alpha charlie", "zzz"], x[:3], np.array([2, 2, 2]), [])
    f = fresh.updates[0]
    assert f["id"] is None and f["size"] == 3 and f["members"] == [0, 1, 2] and f["name"] == "alpha / bravo / charlie"
    assert f["centroid"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert fresh.n_clusters == 1 and fresh.n_noise == 0


def test_noise_uses_merged_centroids_and_clusters_sorted() -> None:
    x = np.asarray([_unit(2), _unit(2), _unit(3), _unit(3), _unit(2)], dtype=np.float32)
    existing = [{"id": 1, "name": "a", "centroid": _unit(0).tolist(), "size": 1}]
    out = assign_categories(list("abcde"), x, np.array([7, 7, 3, 3, -1]), existing, match_threshold=2.0)
    assert [u["members"] for u in out.updates] == [[2, 3], [0, 1]]  # cluster ids processed in sorted order
    assert out.category_ids[4] is None


class _Spy:
    """Records constructor kwargs and inputs of the sklearn estimators cluster_embeddings uses."""

    def __init__(self, real: type, log: list) -> None:
        self.real, self.log = real, log

    def __call__(self, **kw):  # type: ignore[no-untyped-def]
        est = self.real(**kw)
        log = self.log

        class Wrapped:
            def fit_transform(self, x):  # type: ignore[no-untyped-def]
                log.append(("fit_transform", kw, np.array(x)))
                return est.fit_transform(x)

            def fit_predict(self, x):  # type: ignore[no-untyped-def]
                log.append(("fit_predict", kw, np.array(x)))
                return est.fit_predict(x)

        return Wrapped()


@pytest.fixture()
def spy(monkeypatch: pytest.MonkeyPatch) -> list:
    import evalforge.cluster as cl

    log: list = []
    monkeypatch.setattr(cl, "TruncatedSVD", _Spy(cl.TruncatedSVD, log))
    monkeypatch.setattr(cl, "HDBSCAN", _Spy(cl.HDBSCAN, log))
    return log


def test_cluster_estimator_configuration(spy: list) -> None:
    x = np.random.default_rng(0).random((60, 64)).astype(np.float32)
    cluster_embeddings(x)
    (svd_call, svd_kw, _), (hdb_call, hdb_kw, z) = spy
    assert svd_call == "fit_transform" and svd_kw == {"n_components": 48, "random_state": 0}
    assert hdb_call == "fit_predict" and hdb_kw == {"min_cluster_size": 5, "min_samples": 2, "copy": True}
    assert z.shape == (60, 48)
    assert np.linalg.norm(z, axis=1) == pytest.approx(np.ones(60), abs=1e-6)


def test_cluster_svd_dims_capped_by_rows_and_features(spy: list) -> None:
    rng = np.random.default_rng(1)
    cluster_embeddings(rng.random((10, 64)), min_cluster_size=3, seed=7)
    assert spy[0][1] == {"n_components": 9, "random_state": 7}
    spy.clear()
    cluster_embeddings(rng.random((30, 6)), min_cluster_size=3, svd_dims=48)
    assert spy[0][1]["n_components"] == 5
    spy.clear()
    cluster_embeddings(rng.random((30, 64)), min_cluster_size=3, svd_dims=4)
    assert spy[0][1]["n_components"] == 4


def test_cluster_skips_svd_below_two_dims_and_normalises_raw_rows(spy: list) -> None:
    x = np.array([[0.3, 0.4], [0.6, 0.8], [0.0, 0.5], [0.0, 0.25]], dtype=np.float64)
    cluster_embeddings(x, min_cluster_size=2)  # dims = min(48, 3, 1) = 1 -> no SVD
    assert [c[0] for c in spy] == ["fit_predict"]
    z = spy[0][2]
    np.testing.assert_allclose(z, [[0.6, 0.8], [0.6, 0.8], [0.0, 1.0], [0.0, 1.0]], atol=1e-9)
    spy.clear()
    three = np.random.default_rng(2).random((3, 3))  # dims = 2 -> SVD runs
    cluster_embeddings(three, min_cluster_size=2)
    assert [c[0] for c in spy] == ["fit_transform", "fit_predict"] and spy[0][1]["n_components"] == 2


def test_cluster_runs_hdbscan_exactly_at_the_size_guard(spy: list) -> None:
    rng = np.random.default_rng(3)
    cluster_embeddings(rng.random((3, 8)), min_cluster_size=2)  # n == max(3, 2)
    assert spy and spy[-1][0] == "fit_predict"
    spy.clear()
    cluster_embeddings(rng.random((5, 8)), min_cluster_size=5)  # n == min_cluster_size
    assert spy and spy[-1][0] == "fit_predict"
    spy.clear()
    assert cluster_embeddings(rng.random((4, 8))).tolist() == [-1] * 4  # 4 < default 5
    assert spy == []
    cluster_embeddings(rng.random((5, 8)))  # 5 == default 5
    assert spy and spy[-1][1]["min_cluster_size"] == 5


def test_assign_with_no_existing_never_matches_even_at_negative_thresholds() -> None:
    x = np.asarray([_unit(0), _unit(0), _unit(1)], dtype=np.float32)
    out = assign_categories(["a", "b", "c"], x, np.array([0, 0, -1]), [], match_threshold=-1.0, noise_threshold=-1.0)
    assert out.updates[0]["id"] is None and out.category_ids == [None, None, None]

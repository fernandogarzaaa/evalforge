"""Interaction clustering (HDBSCAN) and a persistent category registry.

HDBSCAN is used because the number of interaction types is unknown and because it
labels points that do not belong to any dense region as noise (-1). That noise is the
"novel request" signal for edge-case mining, which k-means cannot give you.

Cluster ids from one nightly run mean nothing the next night, so clusters are mapped
onto stable *categories* by centroid similarity. Coverage and growth are tracked per
category, not per run-local cluster.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from evalforge.embed import cosine_matrix, normalise_text

UNCLUSTERED = "unclustered"
EXTRA_STOP = {
    "hi",
    "hey",
    "hello",
    "thanks",
    "thank",
    "thx",
    "asap",
    "quick",
    "question",
    "team",
    "please",
    "want",
    "acme",
    "does",
    "doesn",
    "don",
    "just",
    "like",
    "need",
    "help",
    "know",
    "tell",
    "make",
    "use",
}


def cluster_embeddings(x: np.ndarray, min_cluster_size: int = 5, svd_dims: int = 48, seed: int = 0) -> np.ndarray:
    """Return an HDBSCAN label per row (-1 = noise). Reduces with SVD first for speed and stability."""
    n = x.shape[0]
    if n < max(3, min_cluster_size):
        return np.full(n, -1, dtype=int)
    dims = min(svd_dims, n - 1, x.shape[1] - 1)
    z = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(x) if dims >= 2 else x
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2, copy=True).fit_predict(z)
    return np.asarray(labels, dtype=int)


def top_terms(texts: Sequence[str], k: int = 3) -> list[str]:
    counts: Counter[str] = Counter()
    for t in texts:
        words = set(re.findall(r"[a-z][a-z0-9_]{2,}", normalise_text(t)))
        counts.update(w for w in words if w not in ENGLISH_STOP_WORDS and w not in EXTRA_STOP)
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def name_cluster(texts: Sequence[str]) -> str:
    terms = top_terms(texts)
    return " / ".join(terms) if terms else "misc"


def representative(texts: Sequence[str], x: np.ndarray) -> str:
    """The member closest to the cluster centroid: a readable example for reviewers."""
    centroid = x.mean(axis=0, keepdims=True)
    return texts[int(np.argmax(cosine_matrix(x, centroid)[:, 0]))]


@dataclass
class Assignment:
    category_ids: list[int | None]  # per input row; None = unclustered
    updates: list[dict[str, Any]]  # categories to persist: {id|None, name, centroid, size, members}
    n_clusters: int
    n_noise: int


def assign_categories(
    texts: Sequence[str],
    x: np.ndarray,
    labels: np.ndarray,
    existing: Sequence[dict[str, Any]],
    match_threshold: float = 0.7,
    noise_threshold: float = 0.8,
) -> Assignment:
    """Map run-local clusters to persistent categories.

    A cluster joins the most similar existing category when centroid cosine >= ``match_threshold``,
    otherwise it founds a new one. Noise points join an existing category only on a strong match
    (>= ``noise_threshold``); the rest stay unclustered and are treated as novel.
    """
    cats = [dict(c) for c in existing]
    cat_centroids = np.asarray([c["centroid"] for c in cats], dtype=np.float32) if cats else np.zeros((0, x.shape[1]))
    ids: list[int | None] = [None] * len(texts)
    updates: list[dict[str, Any]] = []
    cluster_ids = sorted(set(int(v) for v in labels) - {-1})
    for cid in cluster_ids:
        members = [i for i, v in enumerate(labels) if v == cid]
        centroid = x[members].mean(axis=0)
        match = None
        if len(cat_centroids):
            sims = cosine_matrix(centroid[None, :], cat_centroids)[0]
            best = int(np.argmax(sims))
            if sims[best] >= match_threshold:
                match = cats[best]
        if match is None:
            update = {
                "id": None,
                "name": name_cluster([texts[i] for i in members]),
                "centroid": centroid.tolist(),
                "size": len(members),
                "members": members,
            }
        else:
            total = match["size"] + len(members)
            merged = (np.asarray(match["centroid"]) * match["size"] + centroid * len(members)) / total
            match["centroid"], match["size"] = merged.tolist(), total  # later clusters merge into the update
            cat_centroids[best] = merged
            update = {
                "id": match["id"],
                "name": match["name"],
                "centroid": merged.tolist(),
                "size": total,
                "members": members,
            }
        updates.append(update)
    noise = [i for i, v in enumerate(labels) if v == -1]
    if noise and len(cat_centroids):
        sims = cosine_matrix(x[noise], cat_centroids)
        for row, i in enumerate(noise):
            best = int(np.argmax(sims[row]))
            if sims[row, best] >= noise_threshold:
                ids[i] = int(cats[best]["id"])
    return Assignment(ids, updates, len(cluster_ids), len(noise))

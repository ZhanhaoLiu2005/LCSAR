from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RWSResult:
    seed: int
    community: np.ndarray
    scores: np.ndarray
    converged: bool


def _adjacency_array(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if np.any(matrix < 0):
        raise ValueError("adjacency entries must be nonnegative")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _degree_vector(adjacency: np.ndarray) -> np.ndarray:
    degrees = adjacency.sum(axis=1).astype(float)
    degrees[degrees == 0.0] = 1.0
    return degrees


def _gap_threshold(scores: np.ndarray, min_size: int) -> float:
    positive = scores[scores > 0.0]
    if positive.size == 0:
        return 0.0
    ordered = np.sort(positive)[::-1]
    if ordered.size == 1:
        return float(ordered[0])
    eps = 1e-12
    ratios = (ordered[:-1] + eps) / (ordered[1:] + eps)
    for index in np.argsort(ratios)[::-1]:
        threshold = ordered[index]
        if np.count_nonzero(scores >= threshold) >= min_size:
            return float(threshold)
    return float(ordered[-1])


def _top_fraction_threshold(scores: np.ndarray, fraction: float) -> float:
    positive = scores[scores > 0.0]
    if positive.size == 0:
        return 0.0
    ordered = np.sort(positive)[::-1]
    keep = max(1, int(np.ceil(positive.size * fraction)))
    return float(ordered[min(keep - 1, ordered.size - 1)])


def _sweep(scores: np.ndarray, threshold: float) -> np.ndarray:
    swept = scores.copy()
    swept[swept < threshold] = 0.0
    total = swept.sum()
    if total > 0.0:
        swept /= total
    return swept


def detect_local_community(
    adjacency: np.ndarray,
    seed: int,
    max_iter: int = 4,
    min_size: int = 2,
    include_seed: bool = False,
) -> RWSResult:
    adjacency = _adjacency_array(adjacency)
    n_nodes = adjacency.shape[0]
    if seed < 0 or seed >= n_nodes:
        raise IndexError("seed is outside the node index range")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    degrees = _degree_vector(adjacency)
    neighbors = np.flatnonzero(adjacency[seed] > 0.0)
    scores = np.zeros(n_nodes, dtype=float)
    scores[seed] = 1.0

    denom = float(neighbors.size + 1)
    scores = np.zeros(n_nodes, dtype=float)
    scores[seed] = 1.0 / denom
    if neighbors.size:
        scores[neighbors] = 1.0 / denom

    previous = set(np.flatnonzero(scores > 0.0))
    converged = False

    if max_iter == 1:
        scores = _sweep(scores, _top_fraction_threshold(scores, 0.20))
    else:
        for step in range(2, max_iter + 1):
            next_scores = adjacency.T @ (scores / degrees)
            if step == max_iter:
                scores = _sweep(next_scores, _top_fraction_threshold(next_scores, 0.20))
                break

            scores = _sweep(next_scores, _gap_threshold(next_scores, min_size=min_size))
            current = set(np.flatnonzero(scores > 0.0))
            if current == previous:
                converged = True
                break
            previous = current

    mask = scores > 0.0
    if not include_seed:
        mask[seed] = False
    return RWSResult(
        seed=int(seed),
        community=np.flatnonzero(mask).astype(int),
        scores=scores,
        converged=bool(converged),
    )


def rws_weight_matrix(
    adjacency: np.ndarray,
    max_iter: int = 4,
    min_size: int = 2,
    fallback: str = "neighbors",
    return_results: bool = False,
):
    adjacency = _adjacency_array(adjacency)
    n_nodes = adjacency.shape[0]
    weights = np.zeros((n_nodes, n_nodes), dtype=float)
    results: list[RWSResult] = []

    for seed in range(n_nodes):
        result = detect_local_community(
            adjacency,
            seed=seed,
            max_iter=max_iter,
            min_size=min_size,
            include_seed=False,
        )
        community = result.community
        if community.size == 0 and fallback == "neighbors":
            community = np.flatnonzero(adjacency[seed] > 0.0)
        if community.size:
            weights[seed, community] = 1.0 / float(community.size)
        results.append(result)

    if return_results:
        return weights, results
    return weights

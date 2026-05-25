from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SBMGraph:
    adjacency: np.ndarray
    labels: np.ndarray
    degrees: np.ndarray


def generate_sbm(
    n_nodes: int,
    n_blocks: int,
    p_out: float,
    p_ratio: float,
    seed: int | None = None,
    block_probs: np.ndarray | None = None,
) -> SBMGraph:
    if n_nodes <= 0 or n_blocks <= 0:
        raise ValueError("n_nodes and n_blocks must be positive")
    p_in = p_out * p_ratio
    if not (0.0 <= p_out <= 1.0 and 0.0 <= p_in <= 1.0):
        raise ValueError("edge probabilities must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    if block_probs is None:
        probs = np.full(n_blocks, 1.0 / n_blocks)
    else:
        probs = np.asarray(block_probs, dtype=float)
        if probs.size != n_blocks or np.any(probs < 0.0) or probs.sum() <= 0.0:
            raise ValueError("block_probs must be nonnegative and have length n_blocks")
        probs = probs / probs.sum()

    labels = rng.choice(np.arange(n_blocks), size=n_nodes, replace=True, p=probs)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes - 1):
        same_block = labels[i] == labels[(i + 1) :]
        edge_probs = np.where(same_block, p_in, p_out)
        edges = rng.random(n_nodes - i - 1) < edge_probs
        adjacency[i, (i + 1) :] = edges.astype(float)
    adjacency = adjacency + adjacency.T
    np.fill_diagonal(adjacency, 0.0)
    degrees = adjacency.sum(axis=1)
    return SBMGraph(adjacency=adjacency, labels=labels.astype(int), degrees=degrees)


def _row_normalize(mask: np.ndarray) -> np.ndarray:
    matrix = np.asarray(mask, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    row_sums = matrix.sum(axis=1)
    nonzero = row_sums > 0.0
    matrix[nonzero] = matrix[nonzero] / row_sums[nonzero, None]
    return matrix


def oracle_weight_matrix(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    same_block = labels[:, None] == labels[None, :]
    return _row_normalize(same_block.astype(float))


def neighbor_weight_matrix(adjacency: np.ndarray) -> np.ndarray:
    return _row_normalize(np.asarray(adjacency, dtype=float))


def simulate_sar_response(
    weights: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    rho: float,
    sigma: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    design = np.asarray(x, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n_rows = weights.shape[0]
    if weights.shape != (n_rows, n_rows):
        raise ValueError("weights must be square")
    if design.shape[0] != n_rows:
        raise ValueError("X must have the same number of rows as weights")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=n_rows)
    system = np.eye(n_rows) - rho * weights
    signal = design @ beta + noise
    try:
        return np.linalg.solve(system, signal)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ signal

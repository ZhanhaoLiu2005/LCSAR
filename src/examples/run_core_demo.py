from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcsar import fit_lcsar_lse, generate_sbm, oracle_weight_matrix, rws_weight_matrix
from lcsar.simulation import simulate_sar_response


def main() -> None:
    graph = generate_sbm(
        n_nodes=120,
        n_blocks=4,
        p_out=0.04,
        p_ratio=10,
        seed=2026,
    )
    oracle_w = oracle_weight_matrix(graph.labels)
    rws_w = rws_weight_matrix(graph.adjacency, max_iter=4)

    rng = np.random.default_rng(2027)
    x = rng.normal(size=(graph.adjacency.shape[0], 3))
    beta = np.array([1.2, 0.5, -0.3])
    y = simulate_sar_response(oracle_w, x, beta=beta, rho=0.7, sigma=1.0, seed=2028)

    fit = fit_lcsar_lse(y, x, rws_w)
    print(f"rho_hat: {fit.rho:.4f}")
    print(f"t_rho:   {fit.t_rho:.4f}")
    print(f"R2:      {fit.r_squared:.4f}")


if __name__ == "__main__":
    main()

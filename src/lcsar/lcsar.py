from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class LCSARFit:
    rho: float
    beta: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    sigma2: float
    se_rho: float
    t_rho: float
    p_rho: float
    r_squared: float
    adjusted_r_squared: float


def _as_vector(y: np.ndarray) -> np.ndarray:
    vector = np.asarray(y, dtype=float).reshape(-1)
    if vector.ndim != 1:
        raise ValueError("y must be one-dimensional")
    return vector


def _as_design(x: np.ndarray, n_rows: int, add_intercept: bool) -> np.ndarray:
    design = np.asarray(x, dtype=float)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.ndim != 2 or design.shape[0] != n_rows:
        raise ValueError("X must be a two-dimensional array with len(y) rows")
    if add_intercept:
        design = np.column_stack([np.ones(n_rows), design])
    return design


def _as_weight_matrix(w: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.asarray(w, dtype=float)
    if matrix.shape != (n_rows, n_rows):
        raise ValueError("W must be a square matrix with dimension len(y)")
    return matrix


def fit_lcsar_lse(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    add_intercept: bool = False,
    ridge: float = 1e-10,
) -> LCSARFit:
    y_vec = _as_vector(y)
    n_rows = y_vec.size
    design = _as_design(x, n_rows=n_rows, add_intercept=add_intercept)
    weights = _as_weight_matrix(w, n_rows=n_rows)

    spatial_lag = weights @ y_vec
    xtx = design.T @ design
    xtx_inv = np.linalg.pinv(xtx + ridge * np.eye(xtx.shape[0]))

    def residualize(vector: np.ndarray) -> np.ndarray:
        return vector - design @ (xtx_inv @ (design.T @ vector))

    lag_resid = residualize(spatial_lag)
    y_resid = residualize(y_vec)
    denom = float(lag_resid.T @ lag_resid)
    rho = float((lag_resid.T @ y_resid) / denom) if denom > ridge else 0.0

    beta = xtx_inv @ design.T @ (y_vec - rho * spatial_lag)
    fitted = rho * spatial_lag + design @ beta
    residuals = y_vec - fitted

    n_beta = design.shape[1]
    n_params = n_beta + 1
    df_resid = n_rows - n_params
    sse = float(residuals.T @ residuals)
    sigma2 = sse / df_resid if df_resid > 0 else np.nan

    sst = float(((y_vec - y_vec.mean()).T @ (y_vec - y_vec.mean())))
    r_squared = 1.0 - sse / sst if sst > 0.0 else 0.0
    adj_denom = n_rows - n_params - 1
    adjusted = (
        1.0 - (1.0 - r_squared) * (n_rows - 1) / adj_denom
        if adj_denom > 0
        else r_squared
    )

    se_rho = float(np.sqrt(sigma2 / denom)) if denom > ridge and sigma2 >= 0.0 else np.inf
    t_rho = float(rho / se_rho) if np.isfinite(se_rho) and se_rho > 0.0 else 0.0
    p_rho = (
        float(2.0 * stats.t.sf(abs(t_rho), df=df_resid))
        if df_resid > 0 and np.isfinite(t_rho)
        else np.nan
    )

    return LCSARFit(
        rho=rho,
        beta=np.asarray(beta, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        sigma2=float(sigma2),
        se_rho=se_rho,
        t_rho=t_rho,
        p_rho=p_rho,
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted),
    )

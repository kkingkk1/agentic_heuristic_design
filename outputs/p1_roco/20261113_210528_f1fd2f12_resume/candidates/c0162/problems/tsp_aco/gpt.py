import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    symmetric = 0.5 * (
        np.nan_to_num(raw, nan=np.inf, posinf=np.inf, neginf=0.0)
        + np.nan_to_num(raw.T, nan=np.inf, posinf=np.inf, neginf=0.0)
    )
    symmetric = np.maximum(symmetric, 0.0)

    valid = finite_nonnegative & finite_nonnegative.T
    diagonal = np.eye(n, dtype=bool)
    valid &= ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = symmetric[valid & (symmetric > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    distances = np.where(valid, symmetric, np.inf)
    np.fill_diagonal(distances, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + safe_scale * 1e-9)
    inverse /= max(float(np.max(inverse)), eps)

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_rank /= max(float(np.max(mutual_rank)), eps)

    row_mass = inverse.sum(axis=1, keepdims=True)
    col_mass = inverse.sum(axis=0, keepdims=True)
    row_share = inverse / np.maximum(row_mass, eps)
    col_share = inverse / np.maximum(col_mass, eps)
    competitiveness = np.sqrt(row_share * col_share)
    competitiveness /= max(float(np.max(competitiveness)), eps)

    neighborhood_size = min(4, n - 1)
    local_scale = np.full(n, safe_scale, dtype=float)

    for i in range(n):
        neighbors = distances[i, valid[i]]
        if neighbors.size:
            nearest = np.sort(neighbors)[:neighborhood_size]
            local_scale[i] = max(float(np.mean(nearest)), safe_scale * 0.05)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    contrast = np.zeros((n, n), dtype=float)
    contrast[valid] = np.exp(
        -np.minimum(distances[valid] / (pair_scale[valid] + eps), 50.0)
    )
    contrast /= max(float(np.max(contrast)), eps)

    evidence = (
        0.46 * np.log(np.maximum(inverse, eps))
        + 0.24 * np.log(np.maximum(mutual_rank, eps))
        + 0.18 * np.log(np.maximum(competitiveness, eps))
        + 0.12 * np.log(np.maximum(contrast, eps))
    )

    valid_log_distances = np.log1p(distances[valid] / safe_scale)
    median_log = float(np.median(valid_log_distances))
    dispersion = float(
        np.median(np.abs(valid_log_distances - median_log))
    )
    temperature = float(np.clip(0.85 + 0.25 * dispersion, 0.85, 1.30))

    weights = np.zeros((n, n), dtype=float)
    scaled = evidence[valid] / temperature
    offset = float(np.max(scaled))
    weights[valid] = np.exp(np.clip(scaled - offset, -60.0, 0.0))

    exploration = float(np.clip(0.04 + 0.025 * dispersion, 0.04, 0.08))
    valid_count = float(np.sum(valid))
    weights = (
        (1.0 - exploration) * weights
        + exploration * valid.astype(float) / max(valid_count, 1.0)
    )

    weights = 0.5 * (weights + weights.T)

    for _ in range(8):
        row_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(row_sum, eps)
        weights = 0.5 * (weights + weights.T)

    result = 0.72 * weights + 0.28 * np.sqrt(
        np.maximum(inverse * contrast, 0.0)
    )
    result = 0.5 * (result + result.T)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return soft edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    positive = distances[valid & (distances > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe = np.where(valid, np.maximum(distances, 0.0), np.inf)
    inverse = np.zeros_like(safe, dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    col_max = np.max(inverse, axis=0, keepdims=True)

    row_comp = inverse / (row_max + eps)
    col_comp = inverse / (col_max + eps)
    local_rank = np.sqrt(row_comp * col_comp)

    row_counts = np.sum(valid, axis=1)
    col_counts = np.sum(valid, axis=0)
    k_row = max(1, min(3, n - 1))

    row_top = np.partition(
        np.where(valid, inverse, 0.0),
        -k_row,
        axis=1,
    )[:, -k_row:]
    col_top = np.partition(
        np.where(valid, inverse, 0.0),
        -k_row,
        axis=0,
    )[-k_row:, :]

    row_potential = np.sum(row_top, axis=1) / np.maximum(
        np.minimum(row_counts, k_row), 1
    )
    col_potential = np.sum(col_top, axis=0) / np.maximum(
        np.minimum(col_counts, k_row), 1
    )

    node_potential = np.sqrt(
        np.maximum(row_potential, eps) *
        np.maximum(col_potential, eps)
    )
    node_potential /= np.max(node_potential) + eps

    edge_potential = np.sqrt(
        node_potential[:, None] * node_potential[None, :]
    )

    valid_distances = distances[valid]
    log_distances = np.log1p(np.maximum(valid_distances, 0.0) / scale)
    median_log = float(np.median(log_distances))
    dispersion = float(np.median(np.abs(log_distances - median_log)))
    temperature = float(np.clip(0.70 + 0.45 * dispersion, 0.70, 1.35))

    inverse_scaled = inverse / (np.max(inverse) + eps)
    inverse_scaled = np.maximum(inverse_scaled, eps)
    local_rank = np.maximum(local_rank, eps)
    edge_potential = np.maximum(edge_potential, eps)

    log_score = (
        0.60 * np.log(inverse_scaled)
        + 0.22 * np.log(local_rank)
        + 0.18 * np.log(edge_potential)
    )

    masked_score = np.where(valid, log_score / temperature, -np.inf)
    score_max = np.max(masked_score[valid])
    exponential = np.zeros_like(masked_score)
    exponential[valid] = np.exp(masked_score[valid] - score_max)

    valid_count = float(np.sum(valid))
    exploration_floor = float(np.clip(0.06 + 0.04 * (temperature - 0.70), 0.06, 0.10))

    priors = (
        (1.0 - exploration_floor) * exponential
        + exploration_floor * valid.astype(float) / valid_count
    )

    priors[~valid] = 0.0

    maximum = np.max(priors)
    if maximum > 0.0:
        priors /= maximum

    return priors

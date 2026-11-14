import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized soft edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)
    values = symmetric[valid]

    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    distances = np.where(valid, np.maximum(symmetric, 0.0), np.inf)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[valid]
    penalty = float(np.max(finite_values) + 12.0 * safe_scale)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid], penalty) / safe_scale
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + safe_scale * 1e-9)

    inverse /= max(float(np.max(inverse)), eps)

    row_mass = affinity.sum(axis=1)
    col_mass = affinity.sum(axis=0)

    row_share = affinity / np.maximum(row_mass[:, None], eps)
    col_share = affinity / np.maximum(col_mass[None, :], eps)

    mutual_competitiveness = np.sqrt(
        np.maximum(row_share * col_share, eps)
    )

    k = max(1, min(4, n - 1))
    top_row = np.partition(
        np.where(valid, affinity, 0.0),
        -k,
        axis=1,
    )[:, -k:]
    top_col = np.partition(
        np.where(valid, affinity, 0.0),
        -k,
        axis=0,
    )[-k:, :]

    row_endpoint = np.mean(top_row, axis=1)
    col_endpoint = np.mean(top_col, axis=0)
    endpoint_potential = np.sqrt(
        np.maximum(row_endpoint[:, None] * col_endpoint[None, :], eps)
    )
    endpoint_potential /= max(float(np.max(endpoint_potential)), eps)

    row_sorted = np.sort(
        np.where(valid, distances, penalty),
        axis=1,
    )
    best = row_sorted[:, 0]
    second = row_sorted[:, min(1, n - 1)]
    local_gap_scale = float(np.median(np.maximum(second - best, 0.0)))
    local_gap_scale = max(local_gap_scale, safe_scale * 1e-3)

    row_gap = np.maximum(second[:, None] - distances, 0.0)
    col_sorted = np.sort(
        np.where(valid, distances.T, penalty),
        axis=1,
    )
    col_best = col_sorted[:, 0]
    col_second = col_sorted[:, min(1, n - 1)]
    col_gap = np.maximum(col_second[None, :] - distances, 0.0)

    exclusivity = 1.0 - np.exp(
        -0.5 * (row_gap + col_gap) / local_gap_scale
    )
    exclusivity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    for k_node in range(n):
        if k_node == 0:
            candidate = distances[:, k_node, None] + distances[None, k_node, :]
        else:
            candidate = distances[:, k_node, None] + distances[None, k_node, :]
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = penalty
    bypass_advantage = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -bypass_advantage / safe_scale
    )
    irreplaceability[~valid] = 0.0

    log_distance = np.log(np.maximum(inverse, eps))
    log_competitiveness = np.log(
        np.maximum(mutual_competitiveness, eps)
    )
    log_endpoint = np.log(np.maximum(endpoint_potential, eps))
    log_exclusivity = np.log(
        np.maximum(exclusivity, eps)
    )
    log_irreplaceability = np.log(
        np.maximum(irreplaceability, eps)
    )

    score_log = (
        0.42 * log_distance
        + 0.20 * log_competitiveness
        + 0.14 * log_endpoint
        + 0.14 * log_exclusivity
        + 0.10 * log_irreplaceability
    )

    valid_log_distances = np.log1p(
        distances[valid] / safe_scale
    )
    median_log = float(np.median(valid_log_distances))
    dispersion = float(
        np.median(np.abs(valid_log_distances - median_log))
    )
    temperature = float(np.clip(0.75 + 0.35 * dispersion, 0.75, 1.25))

    masked = np.where(valid, score_log / temperature, -np.inf)
    maximum = float(np.max(masked[valid]))

    weights = np.zeros((n, n), dtype=float)
    weights[valid] = np.exp(
        np.clip(masked[valid] - maximum, -60.0, 0.0)
    )

    exploration = float(np.clip(0.05 + 0.03 * dispersion, 0.05, 0.09))
    valid_count = float(np.sum(valid))
    weights = (
        (1.0 - exploration) * weights
        + exploration * valid.astype(float) / valid_count
    )

    weights = 0.5 * (weights + weights.T)
    for _ in range(12):
        row_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(row_sum, eps)
        weights = 0.5 * (weights + weights.T)

    result = 0.7 * weights + 0.3 * np.sqrt(
        np.maximum(inverse * affinity, 0.0)
    )
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

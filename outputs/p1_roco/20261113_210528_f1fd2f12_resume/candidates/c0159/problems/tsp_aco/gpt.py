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

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0

    symmetric_raw = np.where(finite, np.maximum(raw, 0.0), np.nan)
    symmetric_raw = np.nanmean(
        np.stack((symmetric_raw, symmetric_raw.T), axis=0),
        axis=0,
    )

    valid = (
        np.isfinite(symmetric_raw)
        & (symmetric_raw >= 0.0)
        & ~diagonal
    )
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = symmetric_raw[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    finite_max = float(np.max(values)) if values.size else safe_scale
    penalty = max(finite_max + 12.0 * safe_scale, safe_scale)

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = symmetric_raw[valid]
    np.fill_diagonal(distances, penalty)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / safe_scale, 60.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + 1e-9 * safe_scale)
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = np.maximum(affinity.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(affinity.sum(axis=0, keepdims=True), eps)

    row_share = affinity / row_mass
    col_share = affinity / col_mass
    competitiveness = np.sqrt(
        np.maximum(row_share * col_share, eps)
    )

    top_k = min(4, n - 1)
    top_rows = np.partition(
        np.where(valid, affinity, 0.0),
        -top_k,
        axis=1,
    )[:, -top_k:]
    top_cols = np.partition(
        np.where(valid, affinity, 0.0),
        -top_k,
        axis=0,
    )[-top_k:, :]

    endpoint_row = np.mean(top_rows, axis=1)
    endpoint_col = np.mean(top_cols, axis=0)
    endpoint = np.sqrt(
        np.maximum(
            endpoint_row[:, None] * endpoint_col[None, :],
            eps,
        )
    )
    endpoint /= max(float(np.max(endpoint)), eps)

    row_sorted = np.sort(
        np.where(valid, distances, penalty),
        axis=1,
    )
    row_best = row_sorted[:, 0]
    row_second = row_sorted[:, min(1, n - 1)]
    row_gap = np.maximum(row_second[:, None] - distances, 0.0)

    col_sorted = np.sort(
        np.where(valid, distances.T, penalty),
        axis=1,
    )
    col_best = col_sorted[:, 0]
    col_second = col_sorted[:, min(1, n - 1)]
    col_gap = np.maximum(col_second[None, :] - distances, 0.0)

    gap_values = np.concatenate(
        (np.maximum(row_second - row_best, 0.0),
         np.maximum(col_second - col_best, 0.0))
    )
    gap_scale = float(np.median(gap_values))
    gap_scale = max(gap_scale, 0.05 * safe_scale, eps)

    exclusivity = 1.0 - np.exp(
        -0.5 * (row_gap + col_gap) / gap_scale
    )
    exclusivity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    for middle in range(n):
        candidate = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, penalty)

    bypass_advantage = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -bypass_advantage / safe_scale
    )
    irreplaceability[~valid] = 0.0

    log_score = (
        0.44 * np.log(np.maximum(inverse, eps))
        + 0.22 * np.log(np.maximum(competitiveness, eps))
        + 0.14 * np.log(np.maximum(endpoint, eps))
        + 0.12 * np.log(np.maximum(exclusivity, eps))
        + 0.08 * np.log(np.maximum(irreplaceability, eps))
    )

    log_distances = np.log1p(distances[valid] / safe_scale)
    dispersion_center = float(np.median(log_distances))
    dispersion = float(
        np.median(np.abs(log_distances - dispersion_center))
    )
    temperature = float(np.clip(0.85 + 0.30 * dispersion, 0.85, 1.30))

    weights = np.zeros((n, n), dtype=float)
    valid_logits = log_score[valid] / temperature
    maximum = float(np.max(valid_logits))
    weights[valid] = np.exp(
        np.clip(valid_logits - maximum, -60.0, 0.0)
    )

    exploration = float(np.clip(0.04 + 0.03 * dispersion, 0.04, 0.08))
    valid_count = float(np.sum(valid))
    weights = (
        (1.0 - exploration) * weights
        + exploration * valid.astype(float) / valid_count
    )

    weights = 0.5 * (weights + weights.T)

    for _ in range(16):
        row_sum = np.maximum(weights.sum(axis=1, keepdims=True), eps)
        weights /= row_sum
        weights = 0.5 * (weights + weights.T)

    local_component = np.sqrt(
        np.maximum(inverse * affinity, 0.0)
    )
    result = 0.72 * weights + 0.28 * local_component

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
    ).astype(float, copy=False)

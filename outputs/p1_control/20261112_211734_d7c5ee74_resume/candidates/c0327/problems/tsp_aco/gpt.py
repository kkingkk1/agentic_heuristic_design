import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    positive = distances[valid & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    median_positive = float(np.median(positive))
    q25, q75 = np.percentile(positive, [25.0, 75.0])
    spread = max(float(q75 - q25), np.finfo(float).eps)
    scale_floor = max(
        median_positive * 1e-12,
        np.finfo(float).eps,
    )

    upper_fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(upper_fallback) or upper_fallback <= 0.0:
        upper_fallback = float(np.max(positive))
    if not np.isfinite(upper_fallback) or upper_fallback <= 0.0:
        upper_fallback = 1.0

    repaired = np.where(valid, distances, upper_fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)

    work = repaired.copy()
    np.fill_diagonal(work, np.inf)

    finite_values = work[np.isfinite(work)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    row_median = np.median(finite_values.reshape(n, n), axis=1)
    row_median = np.maximum(row_median, scale_floor)

    row_mean = np.mean(finite_values.reshape(n, n), axis=1)
    row_mean = np.maximum(row_mean, scale_floor)

    row_dispersion = np.mean(
        np.abs(finite_values.reshape(n, n) - row_median[:, None]),
        axis=1,
    )
    row_dispersion = np.maximum(row_dispersion, scale_floor)

    endpoint_scale = np.sqrt(
        row_dispersion[:, None] * row_dispersion[None, :]
    )

    distance_center = 0.5 * (
        row_median[:, None] + row_median[None, :]
    )
    distance_center = np.maximum(distance_center, scale_floor)

    edge_distance = np.where(
        np.isfinite(work),
        work,
        upper_fallback,
    )

    global_z = (edge_distance - median_positive) / spread
    global_shortness = 1.0 / (
        1.0 + np.maximum(global_z, 0.0) ** 2
    )

    local_deviation = (
        edge_distance - distance_center
    ) / (
        endpoint_scale + scale_floor
    )
    local_shortness = 1.0 / (
        1.0 + np.maximum(local_deviation, 0.0) ** 2
    )

    endpoint_ratio_i = edge_distance / (
        row_mean[:, None] + row_dispersion[:, None] + scale_floor
    )
    endpoint_ratio_j = edge_distance / (
        row_mean[None, :] + row_dispersion[None, :] + scale_floor
    )

    endpoint_i = 1.0 / (1.0 + np.maximum(endpoint_ratio_i - 0.5, 0.0))
    endpoint_j = 1.0 / (1.0 + np.maximum(endpoint_ratio_j - 0.5, 0.0))
    endpoint_balance = np.sqrt(endpoint_i * endpoint_j)

    scale_ratio = (
        np.maximum(row_dispersion[:, None], scale_floor)
        / np.maximum(row_dispersion[None, :], scale_floor)
    )
    scale_balance = 1.0 / (
        1.0 + 0.20 * np.abs(np.log(scale_ratio))
    )

    absolute_guard = 1.0 / (
        1.0 + np.maximum(
            (edge_distance - median_positive) / (spread + scale_floor),
            0.0,
        )
    )

    score = (
        0.34 * global_shortness
        + 0.28 * local_shortness
        + 0.20 * endpoint_balance
        + 0.10 * scale_balance
        + 0.08 * absolute_guard
    )

    score = np.maximum(score, 0.0)
    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust distribution-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & off_diagonal
    symmetric_valid = valid & valid.T

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    global_median = float(np.median(positive))
    global_mad = float(np.median(np.abs(positive - global_median)))
    scale_floor = max(global_median * 1e-10, global_mad * 1e-10, eps)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    d = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(d, np.nan)

    row_median = np.nanmedian(d, axis=1)
    row_median = np.where(
        np.isfinite(row_median),
        row_median,
        global_median,
    )

    row_mad = np.nanmedian(
        np.abs(d - row_median[:, None]),
        axis=1,
    )
    row_mad = np.where(
        np.isfinite(row_mad),
        row_mad,
        global_mad,
    )
    row_mad = np.maximum(row_mad, scale_floor)

    safe_distance = np.where(np.isfinite(d), d, fallback)

    surplus_i = (
        row_median[:, None] - safe_distance
    ) / (row_mad[:, None] + scale_floor)
    surplus_j = (
        row_median[None, :] - safe_distance
    ) / (row_mad[None, :] + scale_floor)

    bilateral_surplus = 0.5 * (
        np.tanh(0.75 * surplus_i)
        + np.tanh(0.75 * surplus_j)
    )
    surplus_factor = 0.55 + 0.45 * (
        0.5 * (bilateral_surplus + 1.0)
    )

    imbalance = np.abs(surplus_i - surplus_j)
    balance_factor = 0.72 + 0.28 * np.exp(
        -np.clip(0.5 * imbalance, 0.0, 30.0)
    )

    global_scale = max(global_median + global_mad, scale_floor)
    global_kernel = 1.0 / np.square(
        1.0 + np.log1p(safe_distance / global_scale)
    )

    centrality_i = np.exp(
        -np.clip(
            (safe_distance - row_median[:, None])
            / (row_mad[:, None] + scale_floor),
            -30.0,
            30.0,
        )
    )
    centrality_j = np.exp(
        -np.clip(
            (safe_distance - row_median[None, :])
            / (row_mad[None, :] + scale_floor),
            -30.0,
            30.0,
        )
    )
    endpoint_agreement = np.sqrt(centrality_i * centrality_j)
    endpoint_agreement = np.clip(endpoint_agreement, 0.0, 1.0)

    score = (
        global_kernel
        * surplus_factor
        * balance_factor
        * (0.60 + 0.40 * endpoint_agreement)
    )

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

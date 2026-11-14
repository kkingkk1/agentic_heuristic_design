import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative, robust edge-selection priors for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        scale = max(median, 1.4826 * mad, eps)
    else:
        scale = 1.0

    safe_distances = np.where(valid, np.maximum(distances, 0.0), np.inf)
    normalized = np.minimum(safe_distances / scale, 40.0)

    thermal = np.exp(-normalized)
    reciprocal = 1.0 / (1.0 + normalized)
    attraction = 0.68 * thermal + 0.32 * reciprocal
    attraction[~valid] = 0.0

    row_best = np.min(safe_distances, axis=1, keepdims=True)
    col_best = np.min(safe_distances, axis=0, keepdims=True)

    row_gap = np.maximum(safe_distances - row_best, 0.0) / scale
    col_gap = np.maximum(safe_distances - col_best, 0.0) / scale

    competition = 1.0 / (
        1.0
        + 0.42 * np.minimum(row_gap, 8.0)
        + 0.42 * np.minimum(col_gap, 8.0)
    )
    competition[~valid] = 0.0

    base = attraction * competition
    base[~valid] = 0.0

    # Apply a small regret bonus only when a finite second choice exists.
    regret = np.ones((n, n), dtype=float)
    if n >= 3:
        row_second = np.partition(safe_distances, 1, axis=1)[:, 1][:, None]
        col_second = np.partition(safe_distances, 1, axis=0)[1, :][None, :]

        finite_row_second = np.isfinite(row_second)
        finite_col_second = np.isfinite(col_second)

        row_regret = np.maximum(row_second - safe_distances, 0.0) / scale
        col_regret = np.maximum(col_second - safe_distances, 0.0) / scale

        row_regret = np.minimum(row_regret, 6.0)
        col_regret = np.minimum(col_regret, 6.0)

        regret = 1.0 + 0.10 * np.tanh(row_regret + col_regret)
        regret[~(finite_row_second & finite_col_second)] = 1.0
        regret[~valid] = 1.0

    scores = base * regret
    scores[~valid] = 0.0

    # For nearly symmetric inputs, enforce mild bidirectional consistency.
    overlap = valid & valid.T
    if np.any(overlap):
        difference = np.abs(distances - distances.T)
        symmetric_error = float(np.median(difference[overlap]))
        if symmetric_error <= 0.05 * max(scale, eps):
            paired = np.sqrt(
                np.maximum(scores, 0.0) * np.maximum(scores.T, 0.0)
            )
            paired_valid = overlap
            scores[paired_valid] = 0.75 * scores[paired_valid] + 0.25 * paired[paired_valid]

    # Two conservative Sinkhorn-like corrections.
    balanced = scores.copy()
    for _ in range(2):
        row_sum = np.sum(balanced, axis=1, keepdims=True)
        row_factor = 1.0 / np.sqrt(np.maximum(row_sum, eps))

        positive_row_factors = row_factor[np.isfinite(row_factor)]
        row_reference = float(np.median(positive_row_factors))
        row_factor = np.clip(
            row_factor / max(row_reference, eps),
            0.82,
            1.22,
        )
        balanced *= row_factor

        col_sum = np.sum(balanced, axis=0, keepdims=True)
        col_factor = 1.0 / np.sqrt(np.maximum(col_sum, eps))

        positive_col_factors = col_factor[np.isfinite(col_factor)]
        col_reference = float(np.median(positive_col_factors))
        col_factor = np.clip(
            col_factor / max(col_reference, eps),
            0.82,
            1.22,
        )
        balanced *= col_factor
        balanced[~valid] = 0.0

    scores = 0.78 * scores + 0.22 * balanced
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores

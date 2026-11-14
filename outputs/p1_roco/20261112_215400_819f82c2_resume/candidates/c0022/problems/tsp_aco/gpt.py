import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive_offdiag = finite & (d > 0)
    np.fill_diagonal(positive_offdiag, False)

    if not np.any(positive_offdiag):
        return np.zeros_like(d, dtype=float)

    fallback = np.max(d[positive_offdiag])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    safe_d = np.where(
        positive_offdiag,
        d,
        fallback * 1e6,
    )
    np.fill_diagonal(safe_d, fallback * 1e6)

    row_values = np.where(positive_offdiag, d, np.nan)
    row_medians = np.nanmedian(row_values, axis=1)
    row_medians = np.where(
        np.isfinite(row_medians) & (row_medians > 0),
        row_medians,
        fallback,
    )

    scale = np.sqrt(row_medians[:, None] * row_medians[None, :])
    inverse_distance = 1.0 / (1.0 + safe_d / (scale + 1e-12))

    ranks = np.full((n, n), float(n), dtype=float)
    for i in range(n):
        valid = positive_offdiag[i]
        order = np.argsort(np.where(valid, d[i], np.inf), kind="stable")
        valid_order = order[valid[order]]
        ranks[i, valid_order] = np.arange(valid_order.size, dtype=float)

    rank_scale = max(1.0, np.sqrt(float(n)))
    local_preference = np.exp(-((ranks / rank_scale) ** 1.35))
    local_preference[~positive_offdiag] = 0.0

    mutual_preference = np.sqrt(
        local_preference * local_preference.T
    )

    adaptive_scale = np.median(
        safe_d[positive_offdiag]
    ) if np.any(positive_offdiag) else fallback

    closeness = np.exp(-safe_d / (adaptive_scale + 1e-12))
    closeness[~positive_offdiag] = 0.0

    score = (
        0.45 * inverse_distance
        + 0.35 * mutual_preference
        + 0.20 * closeness
    )
    score[~positive_offdiag] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)

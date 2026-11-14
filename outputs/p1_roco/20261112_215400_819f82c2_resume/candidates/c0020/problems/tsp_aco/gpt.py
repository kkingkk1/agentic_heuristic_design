import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic-sampling priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & off_diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe_d = d.copy()
    finite_values = d[valid]
    fallback = float(np.max(finite_values)) if finite_values.size else 1.0
    safe_d[~finite] = fallback * 10.0 + 1.0
    safe_d[np.eye(n, dtype=bool)] = np.inf

    row_min = np.min(np.where(valid, safe_d, np.inf), axis=1)
    row_min[~np.isfinite(row_min)] = 1.0

    row_median = np.median(
        np.where(valid, safe_d, np.nan),
        axis=1
    )
    row_median = np.nan_to_num(row_median, nan=fallback, posinf=fallback)
    local_scale = np.maximum(row_median - row_min, 1e-12)

    relative_cost = (safe_d - row_min[:, None]) / local_scale[:, None]
    shortness = np.exp(-np.clip(relative_cost, 0.0, 30.0))

    preference_from_source = shortness / (
        np.sum(shortness * valid, axis=1, keepdims=True) + 1e-12
    )

    reverse_preference = preference_from_source.T
    mutual_preference = np.sqrt(
        np.maximum(preference_from_source, 0.0)
        * np.maximum(reverse_preference, 0.0)
    )

    ranks = np.argsort(np.argsort(np.where(valid, safe_d, np.inf), axis=1), axis=1)
    rank_quality = np.exp(-ranks / max(1.0, np.sqrt(n)))

    isolation_bonus = 1.0 / np.sqrt(
        np.maximum(row_min[:, None], 1e-12)
        * np.maximum(row_min[None, :], 1e-12)
    )
    isolation_bonus /= np.max(isolation_bonus[valid]) + 1e-12

    scores = (
        0.45 * shortness
        + 0.30 * mutual_preference
        + 0.15 * rank_quality
        + 0.10 * isolation_bonus
    )

    scores *= valid
    scores /= np.max(scores, axis=1, keepdims=True) + 1e-12
    scores[~valid] = 0.0

    return scores.astype(float, copy=False)

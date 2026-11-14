import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0
    d = np.where(np.isfinite(d), d, fallback)

    positive_values = finite_values[finite_values > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(fallback, 1.0)
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative = d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    distance_term = np.exp(-np.clip(relative ** 0.85, 0.0, 50.0))

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    normalized_rank = (
        row_rank / max(float(n - 1), 1.0)
        + row_rank.T / max(float(n - 1), 1.0)
    )
    rank_term = 1.0 / (1.0 + 0.75 * normalized_rank)

    ratio_a = d / local_scale[:, None]
    ratio_b = d / local_scale[None, :]
    local_compatibility = np.exp(
        -0.75
        * (
            np.abs(np.log1p(ratio_a))
            + np.abs(np.log1p(ratio_b))
        )
    )

    global_term = 0.75 + 0.25 * np.exp(
        -np.clip(d / global_scale, 0.0, 50.0)
    )

    score = (
        distance_term
        * rank_term
        * local_compatibility
        * global_term
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

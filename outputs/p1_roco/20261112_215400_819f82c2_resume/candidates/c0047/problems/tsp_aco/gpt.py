import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric distance- and neighborhood-aware TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    # Make the working distance matrix symmetric and safely replace invalid edges.
    safe = np.where(valid, d, global_scale * 1e6)
    safe = 0.5 * (safe + safe.T)
    np.fill_diagonal(safe, np.inf)

    usable = min(3, n - 1)
    sorted_rows = np.sort(safe, axis=1)
    local_scale = np.mean(sorted_rows[:, :usable], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], np.finfo(float).eps)
    )

    relative_shortness = np.exp(
        -np.clip(safe / pair_scale, 0.0, 50.0)
    )

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    nearest = np.argmin(safe, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if np.isfinite(safe[i, j]) and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    degree_pressure = 1.0 + 0.25 * (
        local_scale[:, None] / (local_scale[:, None] + local_scale[None, :])
        + local_scale[None, :] / (local_scale[:, None] + local_scale[None, :])
    )

    score = (
        relative_shortness
        * (0.55 + 0.45 * rank_consensus)
        * (1.0 + 0.20 * mutual)
        * degree_pressure
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

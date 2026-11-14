import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid_values = distances[finite & off_diagonal]

    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = valid_values[valid_values > 0.0]
    if positive_values.size:
        fallback = float(np.max(positive_values))
        base_scale = float(np.median(positive_values))
    else:
        fallback = float(np.max(np.abs(valid_values)))
        base_scale = fallback

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        base_scale = fallback

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    scale = max(base_scale, eps)

    positive_offdiag = d[off_diagonal & (d > 0.0)]
    if positive_offdiag.size:
        spread = float(np.percentile(positive_offdiag, 75.0))
    else:
        spread = scale
    spread = max(spread, scale * 1e-6, eps)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    affinity = np.exp(-np.clip(work / spread, 0.0, 60.0))
    np.fill_diagonal(affinity, 0.0)

    row_mass = np.sum(affinity, axis=1)
    row_mass = np.maximum(row_mass, eps)
    transition = affinity / row_mass[:, None]

    mutual_walk = np.sqrt(
        np.maximum(transition * transition.T, 0.0)
    )

    endpoint_competition = np.sqrt(
        np.maximum(
            affinity / row_mass[:, None]
            * affinity / row_mass[None, :],
            0.0,
        )
    )

    if n > 2:
        detour = work[:, :, None] + work[None, :, :]
        third_node = ~np.eye(n, dtype=bool)
        detour[:, :, ~third_node[0]] = detour[:, :, ~third_node[0]]
        detour_mask = np.ones((n, n, n), dtype=bool)
        indices = np.arange(n)
        detour_mask[indices, indices, :] = False
        detour_mask[:, :, indices] &= (
            np.arange(n)[None, :, None] != np.arange(n)[:, None, None]
        )
        detour = np.where(detour_mask, detour, np.inf)
        best_detour = np.min(detour, axis=2)
    else:
        best_detour = np.full((n, n), 2.0 * fallback, dtype=float)

    shortcut = (best_detour - d) / spread
    shortcut = np.clip(shortcut, -30.0, 30.0)
    shortcut_bonus = 1.0 / (1.0 + np.exp(-shortcut))

    row_best = np.min(work, axis=1)
    row_best = np.maximum(row_best, eps)
    relative_direct = d / np.sqrt(
        row_best[:, None] * row_best[None, :]
    )
    relative_direct = np.clip(relative_direct, 0.0, 60.0)

    direct_preference = np.exp(-np.sqrt(relative_direct))
    direct_preference = np.nan_to_num(
        direct_preference,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    score = (
        shortcut_bonus ** 1.8
        * mutual_walk ** 0.75
        * endpoint_competition ** 0.55
        * direct_preference
    )

    score = 0.5 * (score + score.T)
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

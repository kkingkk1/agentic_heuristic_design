import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive_values = d[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive_values)), eps)
    fallback = max(float(np.max(positive_values)) * 10.0, global_scale)

    safe_d = np.where(valid, d, fallback)
    np.fill_diagonal(safe_d, fallback)

    detour = np.full((n, n), np.inf, dtype=float)

    for intermediary in range(n):
        usable = valid[:, intermediary][:, None] & valid[intermediary, :][None, :]
        candidate = (
            safe_d[:, intermediary][:, None]
            + safe_d[intermediary, :][None, :]
        )
        detour = np.minimum(detour, np.where(usable, candidate, np.inf))

    direct = safe_d
    finite_detour = np.isfinite(detour)

    normalized_direct = direct / global_scale
    shortness = np.exp(-np.minimum(normalized_direct, 60.0))

    margin = np.zeros_like(direct)
    margin[finite_detour] = (
        detour[finite_detour] - direct[finite_detour]
    ) / global_scale
    margin = np.clip(margin, -60.0, 60.0)

    shortcutness = 1.0 / (1.0 + np.exp(-margin))
    shortcutness[~finite_detour] = 1.0

    detour_separation = np.ones_like(direct)
    detour_separation[finite_detour] = 1.0 - np.exp(
        -np.minimum(detour[finite_detour] / global_scale, 60.0)
    )

    score = (
        shortness
        * (0.25 + 0.75 * shortcutness)
        * (0.55 + 0.45 * detour_separation)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

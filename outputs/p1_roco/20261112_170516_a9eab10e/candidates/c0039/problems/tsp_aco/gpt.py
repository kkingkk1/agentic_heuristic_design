import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural edge desirability indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    base = float(np.median(positive)) if positive.size else 1.0
    base = max(base, np.finfo(float).eps)

    finite_values = d[valid]
    replacement = float(np.max(finite_values)) if finite_values.size else base
    replacement = max(replacement, base)

    work = np.where(valid, d, replacement * 4.0)
    np.fill_diagonal(work, np.inf)

    positive_work = work[np.isfinite(work) & (work > 0.0)]
    global_scale = (
        float(np.median(positive_work))
        if positive_work.size
        else base
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    direct = np.where(valid, work, np.inf)

    two_hop = direct[:, :, None] + direct[None, :, :]
    indices = np.arange(n)
    two_hop[:, indices, indices] = np.inf

    best_detour = np.min(two_hop, axis=2)
    ordered_detours = np.partition(two_hop, kth=1 if n > 2 else 0, axis=2)

    if n > 2:
        second_detour = ordered_detours[:, :, 1]
    else:
        second_detour = best_detour * 1.5

    best_detour = np.maximum(best_detour, np.finfo(float).eps)
    second_detour = np.maximum(second_detour, best_detour)

    shortcut_margin = np.maximum(best_detour - direct, 0.0)
    shortcut_ratio = shortcut_margin / (best_detour + global_scale)

    detour_gap = np.maximum(second_detour - best_detour, 0.0)
    uniqueness = detour_gap / (second_detour + global_scale)

    global_prior = 1.0 / (1.0 + direct / global_scale)

    score = (
        0.58 * shortcut_ratio
        + 0.27 * uniqueness
        + 0.15 * global_prior
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

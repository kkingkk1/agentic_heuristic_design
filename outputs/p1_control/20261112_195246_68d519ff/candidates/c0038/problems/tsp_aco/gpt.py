import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    replacement = float(np.median(positive_values))
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = 1.0

    safe = np.where(finite_positive, d, replacement)
    safe = 0.5 * (safe + safe.T)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe) & (safe > 0.0)
    values = safe[valid]
    scale = float(np.median(values)) if values.size else replacement
    spread = float(np.median(np.abs(values - scale))) if values.size else 0.0
    spread = max(spread, np.finfo(float).eps * max(scale, 1.0))

    centered = (safe - scale) / (spread + np.finfo(float).eps)
    global_kernel = 1.0 / (1.0 + np.exp(np.clip(centered, -40.0, 40.0)))
    global_kernel[~valid] = 0.0

    if n > 2:
        two_hop = safe[:, :, None] + safe[None, :, :]
        np.fill_diagonal(two_hop[:, :, 0], np.inf)

        for i in range(n):
            two_hop[i, i, :] = np.inf

        best_detour = np.min(two_hop, axis=2)
        best_detour = np.where(np.isfinite(best_detour), best_detour, replacement)

        margin = (best_detour - safe) / (spread + np.finfo(float).eps)
        detour_term = 1.0 / (1.0 + np.exp(np.clip(-margin, -40.0, 40.0)))
        detour_term[~valid] = 0.0
    else:
        detour_term = np.zeros((n, n), dtype=float)

    score = global_kernel * (0.35 + 0.65 * detour_term)
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

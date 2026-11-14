import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge priorities for stochastic TSP sampling."""
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

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.percentile(positive, 75.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 0.0)

    usable = symmetric_valid.copy()
    if not np.any(usable):
        return np.zeros((n, n), dtype=float)

    temperature = max(scale * 0.65, np.finfo(float).eps)
    kernel = np.exp(
        -np.clip(repaired / temperature, 0.0, 60.0)
    )
    kernel *= usable
    np.fill_diagonal(kernel, 0.0)

    row_mass = np.sum(kernel, axis=1)
    active = row_mass > 0.0
    if not np.any(active):
        return np.zeros((n, n), dtype=float)

    left = np.ones(n, dtype=float)
    right = np.ones(n, dtype=float)

    for _ in range(80):
        denominator_left = kernel @ right
        left[active] = 1.0 / np.maximum(
            denominator_left[active],
            np.finfo(float).tiny,
        )

        denominator_right = kernel.T @ left
        right[active] = 1.0 / np.maximum(
            denominator_right[active],
            np.finfo(float).tiny,
        )

    balanced = left[:, None] * kernel * right[None, :]
    balanced = 0.5 * (balanced + balanced.T)

    if n > 2:
        detour = np.full((n, n), np.inf, dtype=float)

        for intermediary in range(n):
            candidate = (
                repaired[:, intermediary, None]
                + repaired[intermediary, None, :]
            )
            candidate[intermediary, :] = np.inf
            candidate[:, intermediary] = np.inf
            detour = np.minimum(detour, candidate)

        detour = np.where(np.isfinite(detour), detour, 2.0 * scale)
        shortcut = np.maximum(detour - repaired, 0.0)
        shortcut = shortcut / (
            detour + repaired + np.finfo(float).eps
        )
        shortcut = np.clip(shortcut, 0.0, 1.0)
        shortcut_factor = 0.85 + 0.30 * shortcut
    else:
        shortcut_factor = np.ones((n, n), dtype=float)

    score = balanced * shortcut_factor
    score[~usable] = 0.0
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

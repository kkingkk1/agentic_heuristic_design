import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(d)

    if not np.all(finite | diagonal):
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    valid_values = work[np.isfinite(work)]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif valid_values.size:
        scale = float(np.median(valid_values))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)
    valid_edges = np.isfinite(work)
    affinity[valid_edges] = np.exp(
        -np.square(work[valid_edges] / scale)
    )

    np.fill_diagonal(affinity, 0.0)

    degree = np.sum(affinity, axis=1)
    degree_floor = np.finfo(float).eps
    normalized = affinity / np.sqrt(
        np.maximum(degree[:, None], degree_floor)
        * np.maximum(degree[None, :], degree_floor)
    )

    two_step = normalized @ normalized
    two_step[~valid_edges] = 0.0
    np.fill_diagonal(two_step, 0.0)

    closure_scale = np.max(two_step)
    if closure_scale > 0.0:
        closure = two_step / closure_scale
    else:
        closure = np.zeros_like(two_step)

    endpoint_support = np.sqrt(
        np.maximum(degree[:, None], degree_floor)
        * np.maximum(degree[None, :], degree_floor)
    )
    support_scale = np.median(endpoint_support[valid_edges])
    support_balance = 1.0 / (
        1.0 + np.abs(endpoint_support - support_scale) / (support_scale + degree_floor)
    )

    score = normalized
    score *= 0.65 + 0.35 * closure
    score *= 0.75 + 0.25 * support_balance

    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

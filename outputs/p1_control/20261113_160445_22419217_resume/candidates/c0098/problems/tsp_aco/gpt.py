import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally balanced stochastic-sampling prior for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    symmetric_valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(symmetric_valid, False)

    if not np.any(symmetric_valid):
        return result

    symmetric_distance = 0.5 * (
        np.where(finite_nonnegative, distances, 0.0)
        + np.where(finite_nonnegative.T, distances.T, 0.0)
    )
    symmetric_distance = np.where(
        symmetric_valid,
        symmetric_distance,
        np.inf,
    )
    np.fill_diagonal(symmetric_distance, np.inf)

    positive = symmetric_distance[
        np.isfinite(symmetric_distance) & (symmetric_distance > 0.0)
    ]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    normalized = np.clip(symmetric_distance / scale, 0.0, 700.0)

    # Smooth affinity with a sublinear distance transform.
    affinity = np.exp(-np.sqrt(normalized))
    affinity = np.where(symmetric_valid, affinity, 0.0)
    np.fill_diagonal(affinity, 0.0)

    # Entropy-regularized balancing of incident edge mass.
    balanced = affinity.copy()
    for _ in range(40):
        row_sum = balanced.sum(axis=1)
        row_scale = np.where(row_sum > 0.0, 1.0 / row_sum, 0.0)
        balanced *= row_scale[:, None]

        col_sum = balanced.sum(axis=0)
        col_scale = np.where(col_sum > 0.0, 1.0 / col_sum, 0.0)
        balanced *= col_scale[None, :]

    balanced = 0.5 * (balanced + balanced.T)
    balanced = np.where(symmetric_valid, balanced, 0.0)

    # Direct-edge advantage over the best admissible two-hop detour.
    two_hop = (
        symmetric_distance[:, :, None]
        + symmetric_distance[None, :, :]
    )
    best_detour = np.min(two_hop, axis=2)
    valid_detour = np.isfinite(best_detour) & np.isfinite(symmetric_distance)

    savings = np.zeros((n, n), dtype=float)
    savings[valid_detour] = np.maximum(
        best_detour[valid_detour] - symmetric_distance[valid_detour],
        0.0,
    )

    positive_savings = savings[savings > 0.0]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else scale
    )
    savings_scale = max(savings_scale, np.finfo(float).eps)

    detour_bonus = np.tanh(savings / savings_scale)
    detour_bonus = np.where(symmetric_valid, detour_bonus, 0.0)

    score = balanced * (0.72 + 0.28 * detour_bonus)
    score = 0.5 * (score + score.T)
    score = np.where(symmetric_valid, score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)

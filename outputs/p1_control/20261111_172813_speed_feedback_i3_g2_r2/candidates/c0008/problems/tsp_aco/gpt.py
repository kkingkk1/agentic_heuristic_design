import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return normalized edge-priority indicators for stochastic TSP sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(d)
    valid = finite & off_diagonal

    positive = d[finite & (d > 0)]
    if positive.size == 0 or not np.any(valid):
        return score

    scale = float(np.median(positive))
    safe = np.where(valid, np.maximum(d, 0.0), scale * 1e6)

    def normalize(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values, dtype=float)
        selected = values[valid]
        maximum = np.max(selected) if selected.size else 0.0
        if maximum > 0 and np.isfinite(maximum):
            result[valid] = values[valid] / maximum
        return result

    inverse = np.zeros_like(safe, dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + 1e-12)
    inverse = normalize(inverse)

    reciprocal = normalize(0.5 * (inverse + inverse.T))

    ranked = np.partition(safe, kth=1, axis=1)
    nearest_distance = ranked[:, 0]
    second_distance = ranked[:, 1]

    nearest = np.argmin(safe, axis=1)
    has_nearest = np.any(valid, axis=1)
    nearest = np.where(has_nearest, nearest, -1)

    mutual = np.zeros_like(safe, dtype=float)
    source_indices = np.arange(n)
    mutual_mask = (
        valid
        & (nearest[:, None] == source_indices[None, :])
        & (nearest[None, :] == source_indices[:, None])
    )
    mutual[mutual_mask] = 1.0

    # Leave-one-out alternatives for the two endpoints of every edge.
    endpoint_i_alt = np.where(
        nearest[:, None] == source_indices[None, :],
        second_distance[:, None],
        nearest_distance[:, None],
    )
    endpoint_j_alt = endpoint_i_alt.T

    savings_raw = endpoint_i_alt + endpoint_j_alt - safe
    savings = np.maximum(savings_raw, 0.0)
    savings[~valid] = 0.0
    savings[~np.isfinite(savings)] = 0.0
    savings = normalize(savings)

    score = (
        0.50 * inverse
        + 0.20 * reciprocal
        + 0.20 * savings
        + 0.10 * mutual
    )

    score[~valid] = 0.0

    maximum = np.max(score)
    if maximum > 0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative values")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    offdiag = ~np.eye(n, dtype=bool)
    values = d[offdiag]

    q_low, q_high = np.quantile(values, [0.15, 0.85])
    spread = max(float(q_high - q_low), np.finfo(float).eps)

    shortness = np.clip((q_high - d) / spread, 0.0, 1.0)
    np.fill_diagonal(shortness, 0.0)

    endpoint_median = np.median(
        np.where(offdiag, d, np.nan),
        axis=1
    )
    endpoint_median = np.maximum(
        np.nan_to_num(endpoint_median, nan=spread),
        np.finfo(float).eps
    )

    endpoint_competitiveness = np.sqrt(
        np.clip(
            endpoint_median[:, None] / (endpoint_median[:, None] + d),
            0.0,
            1.0
        )
        * np.clip(
            endpoint_median[None, :] / (endpoint_median[None, :] + d),
            0.0,
            1.0
        )
    )
    np.fill_diagonal(endpoint_competitiveness, 0.0)

    two_hop = np.full((n, n), np.inf, dtype=float)
    for middle in range(n):
        candidate = d[:, middle, None] + d[middle, None, :]
        candidate[middle, :] = np.inf
        candidate[:, middle] = np.inf
        two_hop = np.minimum(two_hop, candidate)

    valid_two_hop = np.isfinite(two_hop)
    relative_gain = np.zeros_like(d, dtype=float)
    denominator = two_hop + d + np.finfo(float).eps
    relative_gain[valid_two_hop] = np.clip(
        (two_hop[valid_two_hop] - d[valid_two_hop])
        / denominator[valid_two_hop],
        0.0,
        1.0
    )
    np.fill_diagonal(relative_gain, 0.0)

    score = (
        0.60 * shortness ** 2
        + 0.25 * relative_gain
        + 0.15 * shortness * endpoint_competitiveness
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score

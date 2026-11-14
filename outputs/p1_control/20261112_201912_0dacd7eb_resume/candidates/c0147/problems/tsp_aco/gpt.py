import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return degree-balanced edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~diagonal
    positive_offdiag = d[off_diagonal & (d > 0.0)]

    if positive_offdiag.size:
        center = float(np.median(positive_offdiag))
        q25, q75 = np.percentile(positive_offdiag, [25.0, 75.0])
        spread = max(float(q75 - q25), eps)
    else:
        center = fallback
        spread = max(fallback, 1.0)

    temperature = max(0.5 * center + 0.75 * spread, eps)

    scaled = d / temperature
    kernel = np.exp(-np.clip(scaled ** 1.15, 0.0, 50.0))
    kernel[diagonal] = 0.0
    kernel = np.maximum(kernel, eps)
    kernel[diagonal] = 0.0

    # Symmetric diagonal scaling: make the expected weighted degree
    # approximately equal to the TSP degree target.
    factors = np.ones(n, dtype=float)
    target_degree = 2.0

    for _ in range(80):
        balanced = factors[:, None] * kernel * factors[None, :]
        row_sum = np.sum(balanced, axis=1)
        correction = np.sqrt(
            target_degree / np.maximum(row_sum, eps)
        )
        factors *= correction

        if np.max(np.abs(row_sum - target_degree)) <= 1e-8:
            break

        factors = np.clip(factors, 1e-12, 1e12)

    balanced = factors[:, None] * kernel * factors[None, :]
    balanced = 0.5 * (balanced + balanced.T)
    np.fill_diagonal(balanced, 0.0)

    row_sum = np.sum(balanced, axis=1)
    row_sum = np.maximum(row_sum, eps)

    # Residual degree capacity favors edges that do not consume an
    # endpoint's entire soft degree budget by themselves.
    occupancy = balanced / np.maximum(
        np.sqrt(row_sum[:, None] * row_sum[None, :]),
        eps,
    )
    capacity = np.sqrt(
        np.clip(1.0 - occupancy / target_degree, 0.0, 1.0)
    )
    capacity = 0.60 + 0.40 * capacity

    # A large two-hop value indicates that the direct edge has many
    # strong substitute routes through intermediate vertices.
    two_hop = balanced @ balanced
    np.fill_diagonal(two_hop, 0.0)

    normalized_two_hop = two_hop / np.maximum(
        np.sqrt(row_sum[:, None] * row_sum[None, :]),
        eps,
    )
    substitute_pressure = normalized_two_hop / (
        1.0 + normalized_two_hop
    )
    bridge_factor = 1.0 - 0.35 * substitute_pressure

    score = balanced * capacity * bridge_factor
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

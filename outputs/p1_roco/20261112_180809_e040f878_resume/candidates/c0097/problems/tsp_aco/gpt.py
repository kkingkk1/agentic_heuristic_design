import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priorities based on global cost and two-hop substitutability."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    finite_positive &= ~np.eye(n, dtype=bool)

    if not np.any(finite_positive):
        return scores

    # Keep only edges that are valid in both directions, then form an
    # undirected working matrix.
    symmetric_valid = finite_positive & finite_positive.T
    d = np.full((n, n), np.inf, dtype=float)
    d[symmetric_valid] = 0.5 * (
        distances[symmetric_valid] + distances.T[symmetric_valid]
    )
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return scores

    values = d[valid]
    median = float(np.median(values))
    q1, q3 = np.percentile(values, [25.0, 75.0])
    spread = max(float(q3 - q1), np.finfo(float).eps)
    scale = max(median, spread, np.finfo(float).eps)

    # A global, rank-free preference for short edges.
    normalized_cost = d / scale
    direct_preference = 1.0 / (1.0 + normalized_cost)

    if n > 2:
        # Best two-hop replacement for each edge, excluding its endpoints.
        two_hop = np.full((n, n), np.inf, dtype=float)
        for k in range(n):
            candidate = d[:, k, None] + d[None, k, :]
            two_hop = np.minimum(two_hop, candidate)

        np.fill_diagonal(two_hop, np.inf)

        # How much worse the best two-hop route is than the direct edge.
        detour_ratio = np.divide(
            two_hop,
            d,
            out=np.ones_like(d),
            where=valid & np.isfinite(two_hop),
        )
        detour_excess = np.maximum(detour_ratio - 1.0, 0.0)

        # Saturating response avoids one extreme detour dominating all edges.
        irreplaceability = detour_excess / (1.0 + detour_excess)

        # Aggregate triangle closure around the edge.  Small slack means that
        # the third node offers a geometrically compatible continuation.
        slack = d[:, :, None] - d[:, None, :] - d[None, :, :]
        slack = np.maximum(slack, 0.0)

        compatible = (
            np.isfinite(d[:, :, None])
            & np.isfinite(d[:, None, :])
            & np.isfinite(d[None, :, :])
        )
        np.fill_diagonal(compatible.reshape(n * n, n)[:, 0], False)

        closure_weight = np.exp(-slack / scale) * compatible
        np.fill_diagonal(closure_weight[:, :, 0], 0.0)

        closure_count = np.sum(compatible, axis=2)
        closure_strength = np.divide(
            np.sum(closure_weight, axis=2),
            closure_count,
            out=np.zeros_like(closure_count),
            where=closure_count > 0,
        )

        score = (
            direct_preference
            * (0.65 + 0.35 * irreplaceability)
            * (0.70 + 0.30 * closure_strength)
        )
    else:
        score = direct_preference

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)

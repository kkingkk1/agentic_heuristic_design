import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized TSP edge-priority matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    finite_nonnegative &= ~diagonal

    if not np.any(finite_nonnegative):
        return result

    reverse = finite_nonnegative.T
    usable = finite_nonnegative | reverse

    symmetric = np.full((n, n), np.inf, dtype=float)

    both = finite_nonnegative & reverse
    only_forward = finite_nonnegative & ~reverse
    only_reverse = reverse & ~finite_nonnegative

    symmetric[both] = 0.5 * (
        distances[both] + distances.T[both]
    )
    symmetric[only_forward] = distances[only_forward]
    symmetric[only_reverse] = distances.T[only_reverse]

    np.fill_diagonal(symmetric, np.inf)
    valid = np.isfinite(symmetric) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    edge_values = symmetric[np.triu(valid, k=1)]

    if edge_values.size == 0:
        return result

    median_distance = float(np.median(edge_values))
    q25, q75 = np.percentile(edge_values, [25.0, 75.0])
    spread = max(float(q75 - q25), eps)
    scale = max(median_distance, eps)

    # Smooth global preference for short edges.
    centered = (symmetric - median_distance) / spread
    centered = np.clip(centered, -40.0, 40.0)
    shortness = 1.0 / (1.0 + np.exp(centered))

    # Best two-hop replacement path through an intermediate node.
    best_detour = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left = symmetric[:, k][:, None]
        right = symmetric[k, :][None, :]
        candidate = left + right
        candidate[~np.isfinite(candidate)] = np.inf
        best_detour = np.minimum(best_detour, candidate)

    # A large positive margin means the direct edge is difficult to replace.
    margin = best_detour - symmetric
    finite_detour = np.isfinite(best_detour)

    normalized_margin = np.zeros((n, n), dtype=float)
    usable_margin = finite_detour & valid
    normalized_margin[usable_margin] = (
        np.maximum(margin[usable_margin], 0.0)
        / (
            best_detour[usable_margin]
            + symmetric[usable_margin]
            + scale
        )
    )
    normalized_margin = np.clip(normalized_margin, 0.0, 1.0)

    # Give edges without any finite two-hop alternative a bounded bridge bonus.
    no_detour_bonus = (
        valid & ~finite_detour
    ).astype(float)

    bridge_value = (
        0.35 * normalized_margin
        + 0.20 * no_detour_bonus
    )

    scores = shortness * (1.0 + bridge_value)
    scores[~valid] = 0.0

    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)

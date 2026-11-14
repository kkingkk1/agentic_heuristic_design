import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced and endpoint-exclusive edge priorities for TSP sampling."""
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

    finite_values = distances[finite]
    finite_values = finite_values[finite_values >= 0.0]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    positive_d = d[np.isfinite(d) & (d > 0.0)]
    scale = (
        float(np.median(positive_d))
        if positive_d.size
        else fallback
    )
    scale = max(scale, eps)

    # A bounded inverse-distance kernel with a logarithmic tail.
    normalized = d / scale
    affinity = 1.0 / (1.0 + np.log1p(np.maximum(normalized, 0.0)))
    affinity = np.nan_to_num(
        affinity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(affinity, 0.0)

    # Symmetric Sinkhorn balancing removes node-degree bias.
    balanced = affinity.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1)
        factors = 1.0 / np.sqrt(np.maximum(row_sum, eps))
        balanced *= factors[:, None]
        balanced *= factors[None, :]
        balanced = 0.5 * (balanced + balanced.T)
        np.fill_diagonal(balanced, 0.0)

    # Endpoint exclusivity: measure each edge against the full soft
    # choice distribution of both endpoints rather than using hard ranks.
    row_total = np.maximum(balanced.sum(axis=1), eps)
    share = balanced / row_total[:, None]

    row_sq = np.sum(share * share, axis=1)
    effective_choices = 1.0 / np.maximum(row_sq, eps)
    exclusivity_i = share / np.maximum(
        1.0 / effective_choices[:, None],
        eps,
    )
    exclusivity_j = share.T / np.maximum(
        1.0 / effective_choices[None, :],
        eps,
    )

    endpoint_exclusivity = np.sqrt(
        np.clip(exclusivity_i, 0.0, None)
        * np.clip(exclusivity_j, 0.0, None)
    )
    endpoint_exclusivity = np.sqrt(
        np.clip(endpoint_exclusivity, 0.0, None)
    )

    # Retain a weak global safeguard so isolated long-range links are
    # not completely erased by local balancing.
    global_guard = 0.35 + 0.65 / (
        1.0 + np.sqrt(np.maximum(normalized, 0.0))
    )

    score = (
        np.sqrt(np.maximum(balanced, 0.0))
        * (0.65 + 0.35 * endpoint_exclusivity)
        * global_guard
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    score = np.clip(score, 0.0, 1.0)
    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural edge indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    positive_values = symmetric_d[valid]
    eps = np.finfo(float).eps

    median_distance = max(float(np.median(positive_values)), eps)
    largest_distance = max(float(np.max(positive_values)), median_distance)
    replacement = max(12.0 * largest_distance, median_distance)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # Convert distances into row-wise transition distributions.
    centered = work / median_distance
    row_median = np.median(
        np.where(valid, symmetric_d, np.nan),
        axis=1,
    )
    row_median = np.nan_to_num(
        row_median,
        nan=median_distance,
        posinf=largest_distance,
        neginf=median_distance,
    )
    row_median = np.maximum(row_median, eps)

    local_temperature = np.sqrt(
        row_median[:, None] * row_median[None, :]
    )
    affinity = np.exp(
        -np.minimum(work / np.maximum(local_temperature, eps), 60.0)
    )
    affinity[~valid] = 0.0

    row_sums = np.sum(affinity, axis=1, keepdims=True)
    transition = affinity / np.maximum(row_sums, eps)

    # Entropy measures whether a node has a concentrated or diffuse neighborhood.
    entropy = -np.sum(
        transition * np.log(np.maximum(transition, eps)),
        axis=1,
    )
    entropy /= np.log(max(n - 1, 2))
    concentration = 1.0 - np.clip(entropy, 0.0, 1.0)

    # Reciprocal transition support: both endpoints should favor one another.
    reciprocal_support = np.sqrt(
        np.maximum(transition * transition.T, 0.0)
    )

    # Shared-neighborhood support based on distributional overlap.
    shared_support = np.sqrt(
        np.maximum(transition @ transition.T, 0.0)
    )
    shared_support = shared_support / np.maximum(
        np.sqrt(
            np.sum(transition * transition, axis=1)[:, None]
            * np.sum(transition * transition, axis=1)[None, :]
        ),
        eps,
    )

    # A triangle score rewards edges embedded in short, coherent alternatives.
    triangle_kernel = np.sqrt(np.maximum(affinity, 0.0))
    triangle_mass = triangle_kernel @ triangle_kernel
    triangle_scale = np.maximum(
        np.sqrt(
            np.sum(triangle_kernel, axis=1)[:, None]
            * np.sum(triangle_kernel, axis=1)[None, :]
        ),
        eps,
    )
    triangle_support = triangle_mass / triangle_scale

    # Penalize fragile bridge-like edges whose endpoints have dissimilar neighborhoods.
    neighborhood_gap = np.abs(
        entropy[:, None] - entropy[None, :]
    )
    bridge_penalty = np.exp(-0.85 * neighborhood_gap)

    # Adaptive weighting gives concentrated endpoints more influence.
    endpoint_weight = 0.55 + 0.45 * np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    score = (
        np.power(np.maximum(reciprocal_support, eps), 0.90)
        * np.power(np.maximum(shared_support, eps), 0.75)
        * np.power(np.maximum(triangle_support, eps), 0.60)
        * np.power(np.maximum(bridge_penalty, eps), 0.55)
        * endpoint_weight
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

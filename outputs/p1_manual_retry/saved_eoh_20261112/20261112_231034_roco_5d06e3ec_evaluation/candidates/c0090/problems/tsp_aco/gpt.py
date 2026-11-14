import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric competition- and closure-aware TSP edge prior."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    diagonal = np.eye(n, dtype=bool)

    # Treat an undirected edge as usable only when both directions are valid.
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distance = np.zeros((n, n), dtype=float)
    symmetric_distance[valid] = 0.5 * (
        d[valid] + d.T[valid]
    )

    values = symmetric_distance[valid]
    q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    spread = max(float(q75 - q25), np.finfo(float).eps)

    # Smooth global utility: values below the median receive larger utility.
    centered = (symmetric_distance - q50) / spread
    utility = 1.0 / (1.0 + np.exp(np.clip(1.35 * centered, -60.0, 60.0)))
    utility[~valid] = 0.0
    utility[diagonal] = 0.0

    # Temperature adapts only through each node's available utility mass.
    utility_mass = np.sum(utility, axis=1)
    positive_mass = utility_mass[utility_mass > 0.0]
    mass_reference = (
        float(np.median(positive_mass))
        if positive_mass.size
        else 1.0
    )
    mass_reference = max(mass_reference, np.finfo(float).eps)

    temperature = 0.35 + 0.65 * np.clip(
        utility_mass / mass_reference,
        0.25,
        4.0,
    )

    # Convert utilities into endpoint competition probabilities.
    logits = np.full((n, n), -np.inf, dtype=float)
    logits[valid] = utility[valid] / temperature[:, None][valid]
    row_max = np.max(np.where(valid, logits, -np.inf), axis=1)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)

    weights = np.zeros((n, n), dtype=float)
    weights[valid] = np.exp(
        np.clip(logits[valid] - row_max[:, None][valid], -60.0, 60.0)
    )

    row_sum = np.sum(weights, axis=1)
    row_sum = np.maximum(row_sum, np.finfo(float).eps)
    probabilities = weights / row_sum[:, None]

    # Symmetric endpoint agreement without using neighbor ranks.
    agreement = np.sqrt(
        np.maximum(probabilities * probabilities.T, 0.0)
    )

    # Second-order support: common intermediate nodes with strong endpoint links.
    link_strength = utility / np.maximum(
        utility.sum(axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    closure = link_strength @ link_strength.T
    closure = np.sqrt(np.maximum(closure, 0.0))

    score = (
        np.sqrt(np.maximum(utility, 0.0))
        * (0.30 + 0.70 * agreement)
        * (1.0 + 0.20 * np.clip(closure, 0.0, 1.0))
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

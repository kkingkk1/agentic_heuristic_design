import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return competition-aware edge indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    valid = finite & finite.T & (d >= 0.0) & (d.T >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.zeros((n, n), dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])

    eps = np.finfo(float).eps
    positive_values = symmetric[valid & (symmetric > 0.0)]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0
    scale = max(scale, eps)

    row_values = np.where(valid, symmetric, np.inf)
    finite_max = float(np.max(symmetric[valid])) if np.any(valid) else scale
    finite_max = max(finite_max, scale)

    # A robust temperature prevents both distance concentration and underflow.
    temperature = max(0.35 * scale, eps)
    exponent = -np.minimum(symmetric / temperature, 700.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(exponent[valid])

    # Nodes without usable edges do not participate in balancing.
    active = np.any(valid, axis=1)
    active_indices = np.flatnonzero(active)

    if active_indices.size == 0:
        return np.zeros((n, n), dtype=float)

    weights = np.ones(n, dtype=float)
    weights[~active] = 0.0

    # Symmetric iterative scaling approximates a doubly stochastic edge allocation.
    for _ in range(160):
        row_mass = affinity @ weights
        safe_mass = np.maximum(row_mass, eps)

        updated = np.zeros(n, dtype=float)
        updated[active] = 1.0 / safe_mass[active]
        updated[active] = np.sqrt(
            np.maximum(weights[active], eps) * updated[active]
        )
        updated = np.clip(updated, 1e-150, 1e150)
        updated[~active] = 0.0

        if np.max(np.abs(np.log(
            np.maximum(updated[active], eps)
            / np.maximum(weights[active], eps)
        ))) < 1e-10:
            weights = updated
            break

        weights = updated

    balanced = affinity * weights[:, None] * weights[None, :]
    balanced[~valid] = 0.0
    balanced = 0.5 * (balanced + balanced.T)

    row_mass = np.sum(balanced, axis=1)
    probabilities = np.zeros_like(balanced)
    usable_rows = row_mass > eps
    probabilities[usable_rows] = (
        balanced[usable_rows]
        / row_mass[usable_rows, None]
    )

    # Measure how decisive each node's balanced edge distribution is.
    degree = np.sum(valid, axis=1).astype(float)
    entropy = np.zeros(n, dtype=float)

    for i in active_indices:
        if degree[i] <= 1.0:
            continue
        p = probabilities[i, valid[i]]
        p = p[p > 0.0]
        if p.size:
            entropy[i] = -float(np.sum(p * np.log(p)))
            entropy[i] /= np.log(degree[i])

    certainty = np.clip(1.0 - entropy, 0.0, 1.0)
    endpoint_certainty = np.sqrt(
        (0.5 + 0.5 * certainty[:, None])
        * (0.5 + 0.5 * certainty[None, :])
    )

    score = balanced * endpoint_certainty
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

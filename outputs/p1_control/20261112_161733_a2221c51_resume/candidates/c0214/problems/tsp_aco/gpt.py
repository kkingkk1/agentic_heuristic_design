import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, redundancy-aware edge indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite = np.isfinite(d)
    valid = finite & finite.T & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    positive_values = symmetric_d[valid & (symmetric_d > 0.0)]

    if positive_values.size:
        temperature = float(np.median(positive_values))
    else:
        temperature = 1.0

    temperature = max(temperature, eps)

    safe_distance = np.where(valid, np.maximum(symmetric_d, 0.0), np.inf)
    affinity = np.zeros((n, n), dtype=float)

    finite_rows = np.any(valid, axis=1)
    for i in range(n):
        if not finite_rows[i]:
            continue

        indices = np.flatnonzero(valid[i])
        logits = -safe_distance[i, indices] / temperature
        logits -= np.max(logits)

        weights = np.exp(np.clip(logits, -745.0, 0.0))
        denominator = float(np.sum(weights))

        if denominator > 0.0 and np.isfinite(denominator):
            affinity[i, indices] = weights / denominator

    mutual_affinity = np.sqrt(
        np.maximum(affinity, 0.0) * np.maximum(affinity.T, 0.0)
    )
    mutual_affinity[~valid] = 0.0
    np.fill_diagonal(mutual_affinity, 0.0)

    # Symmetric balancing prevents high-degree endpoints from absorbing
    # disproportionate heuristic mass.
    scaling = np.ones(n, dtype=float)
    for _ in range(32):
        row_mass = mutual_affinity @ scaling
        updated = np.ones(n, dtype=float)
        usable = row_mass > eps
        updated[usable] = 1.0 / np.sqrt(row_mass[usable])
        updated = np.clip(updated, 1e-12, 1e12)

        if np.max(np.abs(updated - scaling)) < 1e-10:
            scaling = updated
            break

        scaling = 0.5 * scaling + 0.5 * updated

    balanced = (
        scaling[:, None] * mutual_affinity * scaling[None, :]
    )
    balanced[~valid] = 0.0

    # Estimate how replaceable each edge is through indirect two-step routes.
    direct_affinity = np.exp(
        np.clip(-np.where(valid, safe_distance, 0.0) / temperature, -745.0, 0.0)
    )
    direct_affinity[~valid] = 0.0

    two_hop = np.zeros((n, n), dtype=float)
    for k in range(n):
        if not finite_rows[k]:
            continue
        left = affinity[:, k]
        right = affinity[k, :]
        two_hop += np.outer(left, right)

    two_hop[~valid] = 0.0

    alternatives = two_hop / np.maximum(
        direct_affinity + two_hop, eps
    )
    alternatives = np.clip(alternatives, 0.0, 1.0)

    score = balanced * (
        0.65 + 0.35 * (1.0 - alternatives)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

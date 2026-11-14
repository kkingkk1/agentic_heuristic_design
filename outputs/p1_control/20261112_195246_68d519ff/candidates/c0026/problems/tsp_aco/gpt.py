import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    scale = np.median(finite_values) if finite_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    z = np.clip(d / scale, 0.0, 700.0)

    inv = 1.0 / (d + scale * 1e-12)
    inv[np.isinf(d)] = 0.0
    inv /= max(np.max(inv), np.finfo(float).eps)

    temperature = max(0.15, float(np.median(z[np.isfinite(z)])))
    logits = -z / temperature

    row_logits = logits - np.max(logits, axis=1, keepdims=True)
    row_affinity = np.exp(np.clip(row_logits, -700.0, 0.0))
    row_affinity /= np.maximum(
        np.sum(row_affinity, axis=1, keepdims=True),
        np.finfo(float).eps,
    )

    col_logits = logits - np.max(logits, axis=0, keepdims=True)
    col_affinity = np.exp(np.clip(col_logits, -700.0, 0.0))
    col_affinity /= np.maximum(
        np.sum(col_affinity, axis=0, keepdims=True),
        np.finfo(float).eps,
    )

    local_consensus = np.sqrt(row_affinity * col_affinity)
    local_consensus = 0.5 * (local_consensus + local_consensus.T)

    nearest = np.argmin(d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    sorted_distances = np.sort(d, axis=1)
    first = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else first
    regret = (second - first) / (second + scale * 1e-12)
    regret = np.clip(regret, 0.0, 1.0)

    endpoint_value = np.sqrt(
        np.maximum(row_affinity.max(axis=1), 0.0)
        * np.maximum(col_affinity.max(axis=0), 0.0)
    )
    endpoint_value = 0.5 * endpoint_value + 0.5 * regret

    bridge = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            via = d[i] + d[:, j]
            via[i] = np.inf
            via[j] = np.inf
            best_detour = np.min(via)
            direct = d[i, j]
            if np.isfinite(best_detour) and np.isfinite(direct):
                bridge_value = best_detour / (direct + best_detour + scale * 1e-12)
                bridge[i, j] = bridge_value
                bridge[j, i] = bridge_value

    prior = (
        0.38 * inv
        + 0.30 * local_consensus
        + 0.18 * mutual
        + 0.09 * bridge
        + 0.05 * (endpoint_value[:, None] + endpoint_value[None, :])
    )

    prior = 0.5 * (prior + prior.T)
    np.fill_diagonal(prior, 0.0)

    maximum = np.max(prior)
    if maximum > 0:
        prior /= maximum

    return prior

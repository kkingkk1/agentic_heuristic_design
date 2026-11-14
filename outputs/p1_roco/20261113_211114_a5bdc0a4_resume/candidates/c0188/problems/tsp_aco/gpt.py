import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & offdiag
    undirected_valid = finite | finite.T

    if not np.any(undirected_valid):
        return np.zeros((n, n), dtype=float)

    # Symmetrize while preserving one-sided finite entries.
    nonnegative = np.maximum(raw, 0.0)
    distances = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T

    distances[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )
    distances[only_forward] = nonnegative[only_forward]
    distances[only_backward] = nonnegative[only_backward]
    distances = 0.5 * (distances + distances.T)
    distances[~undirected_valid] = 0.0
    np.fill_diagonal(distances, 0.0)

    observed = distances[undirected_valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Robust distance affinity.
    temperature = max(0.45 * scale, np.finfo(float).eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[undirected_valid] = np.exp(
        -distances[undirected_valid] / temperature
    )

    # Per-endpoint nearest and second-nearest distances.
    nearest = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)

    for i in range(n):
        row = distances[i].copy()
        row[~undirected_valid[i]] = np.inf
        row[i] = np.inf
        ordered = np.sort(row)
        if ordered.size:
            nearest[i] = ordered[0]
        if ordered.size > 1:
            second[i] = ordered[1]

    local_scale = np.maximum(second - nearest, 0.20 * scale)
    local_scale[~np.isfinite(local_scale)] = 0.20 * scale
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # High values indicate that an edge is especially important to an endpoint.
    endpoint = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid_row = undirected_valid[i]
        endpoint[i, valid_row] = np.exp(
            np.clip(
                (second[i] - distances[i, valid_row]) / local_scale[i],
                -12.0,
                12.0,
            )
        )

    endpoint = np.sqrt(endpoint * endpoint.T)
    endpoint[~undirected_valid] = 0.0

    # Mutual-nearest-neighbor protection.
    nearest_index = np.full(n, -1, dtype=int)
    for i in range(n):
        row = distances[i].copy()
        row[~undirected_valid[i]] = np.inf
        row[i] = np.inf
        if np.any(np.isfinite(row)):
            nearest_index[i] = int(np.argmin(row))

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_index[i]
        if j >= 0 and nearest_index[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Estimate two-hop replaceability using only a small nearest-neighbor set.
    k = min(8, max(1, n - 1))
    candidate_indices = np.full((n, k), -1, dtype=int)

    for i in range(n):
        row = distances[i].copy()
        row[~undirected_valid[i]] = np.inf
        row[i] = np.inf
        order = np.argsort(row, kind="stable")
        finite_order = order[np.isfinite(row[order])]
        take = min(k, finite_order.size)
        if take:
            candidate_indices[i, :take] = finite_order[:take]

    best_bypass = np.full((n, n), np.inf, dtype=float)
    for t in range(k):
        middle = candidate_indices[:, t]
        active = middle >= 0
        if not np.any(active):
            continue

        for m in np.flatnonzero(active):
            left = distances[:, m][:, None]
            right = distances[m, :][None, :]
            route = left + right
            feasible = undirected_valid[:, m][:, None] & undirected_valid[m, :][None, :]
            best_bypass = np.minimum(
                best_bypass,
                np.where(feasible, route, np.inf),
            )

    bypass_gain = np.maximum(best_bypass - distances, 0.0)
    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[undirected_valid] = 1.0 - np.exp(
        -bypass_gain[undirected_valid] / max(0.85 * scale, np.finfo(float).eps)
    )

    affinity_max = float(np.max(affinity))
    base = affinity / max(affinity_max, np.finfo(float).eps)

    score = (
        0.56 * base
        + 0.24 * endpoint
        + 0.12 * irreplaceability
        + 0.08 * mutual
    )
    score[~undirected_valid] = 0.0
    score = 0.5 * (score + score.T)

    # Mild degree correction, avoiding repeated Sinkhorn balancing.
    degree = score.sum(axis=1)
    positive_degree = degree[degree > 0.0]
    reference = (
        float(np.median(positive_degree))
        if positive_degree.size
        else 1.0
    )
    correction = np.sqrt(
        reference / np.maximum(degree[:, None], np.finfo(float).eps)
    )
    corrected = score * correction * correction.T

    result = 0.78 * score + 0.22 * corrected
    result[~undirected_valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

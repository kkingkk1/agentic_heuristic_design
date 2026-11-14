import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic symmetric edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(symmetric, np.inf)

    values = symmetric[valid]
    positive = values[values > 0.0]

    q25, q50, q75, q99 = np.quantile(
        values,
        [0.25, 0.50, 0.75, 0.99],
    )
    spread = float(q75 - q25)

    if np.isfinite(spread) and spread > 0.0:
        scale = spread
    elif np.isfinite(q50) and q50 > 0.0:
        scale = float(q50)
    elif positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(values))
    replacement = max(float(q99), maximum, scale) + 16.0 * scale

    clean = np.where(valid, symmetric, replacement)
    np.fill_diagonal(clean, replacement)

    eps = max(np.finfo(float).eps * scale, 1e-12)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (clean[valid] + eps)

    row_mass = np.maximum(
        inverse.sum(axis=1, keepdims=True),
        eps,
    )
    col_mass = np.maximum(
        inverse.sum(axis=0, keepdims=True),
        eps,
    )

    row_probability = inverse / row_mass
    col_probability = inverse / col_mass
    mutual_preference = np.sqrt(
        np.maximum(row_probability * col_probability, 0.0)
    )

    row_max = np.maximum(
        inverse.max(axis=1, keepdims=True),
        eps,
    )
    col_max = np.maximum(
        inverse.max(axis=0, keepdims=True),
        eps,
    )
    relative_strength = np.sqrt(
        np.maximum(
            (inverse / row_max) * (inverse / col_max),
            0.0,
        )
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(
        n,
        dtype=float,
    )

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~valid] = 0.0

    affinity_temperature = max(0.55 * scale, eps)
    distance_affinity = np.exp(
        -np.minimum(clean / affinity_temperature, 60.0)
    )
    distance_affinity[~valid] = 0.0

    nearest = np.min(clean, axis=1)
    second = np.partition(clean, 1, axis=1)[:, 1]
    local_gap = np.maximum(
        second - nearest,
        0.16 * scale,
    )
    local_gap = np.maximum(local_gap, eps)

    local_preference = np.exp(
        np.clip(
            -(clean - nearest[:, None]) / local_gap[:, None],
            -24.0,
            0.0,
        )
    )
    local_preference[~valid] = 0.0

    preference_max = np.maximum(
        local_preference.max(axis=1, keepdims=True),
        eps,
    )
    local_preference /= preference_max

    endpoint_competition = np.sqrt(
        np.maximum(
            local_preference * local_preference.T,
            0.0,
        )
    )

    endpoint_strength = np.sqrt(
        np.maximum(
            relative_strength * relative_strength.T,
            0.0,
        )
    )

    base = (
        0.26 * endpoint_strength
        + 0.25 * mutual_preference
        + 0.19 * rank_score
        + 0.18 * distance_affinity
        + 0.12 * endpoint_competition
    )
    base[~valid] = 0.0

    best_two_hop = np.full((n, n), replacement, dtype=float)
    block_size = max(
        1,
        min(n, 2_000_000 // max(n, 1)),
    )

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via_block = (
            clean[:, start:stop, None]
            + clean[None, start:stop, :]
        )
        best_two_hop = np.minimum(
            best_two_hop,
            np.min(via_block, axis=1),
        )

    np.fill_diagonal(best_two_hop, replacement)

    bypass_gain = np.maximum(
        best_two_hop - clean,
        0.0,
    )
    bypass_scale = max(0.72 * scale, eps)
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / bypass_scale
    )
    irreplaceability[~valid] = 0.0

    tree_support = np.full((n, n), 0.62, dtype=float)
    tree_support[~valid] = 0.0

    visited = np.zeros(n, dtype=bool)
    best_edge = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for component_start in range(n):
        if visited[component_start]:
            continue

        best_edge[component_start] = 0.0

        while True:
            candidates = np.where(
                ~visited,
                best_edge,
                np.inf,
            )
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            visited[node] = True

            if parent[node] >= 0:
                left = node
                right = parent[node]
                tree_support[left, right] = 1.0
                tree_support[right, left] = 1.0

            neighbors = np.flatnonzero(
                valid[node] & ~visited
            )

            for neighbor in neighbors:
                distance = clean[node, neighbor]
                if distance < best_edge[neighbor]:
                    best_edge[neighbor] = distance
                    parent[neighbor] = node

    score = (
        0.40 * base
        + 0.24 * endpoint_competition
        + 0.20 * irreplaceability
        + 0.16 * tree_support
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)

    balanced = score.copy()

    for _ in range(32):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum

        col_sum = np.maximum(
            balanced.sum(axis=0, keepdims=True),
            eps,
        )
        balanced /= col_sum

    balanced = 0.5 * (balanced + balanced.T)
    balanced[~valid] = 0.0

    result = 0.64 * score + 0.36 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

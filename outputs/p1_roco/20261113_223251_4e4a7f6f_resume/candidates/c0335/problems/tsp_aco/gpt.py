import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    good = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    valid = good | good.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)

    both = good & good.T
    forward_only = good & ~good.T
    backward_only = ~good & good.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]

    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        median = float(np.median(positive))
        q25, q75 = np.quantile(positive, [0.25, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = median
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    safe_scale = max(scale, eps)
    upper = float(np.quantile(observed, 0.90))
    replacement = max(upper, float(np.max(observed)), safe_scale) + 12.0 * safe_scale

    distances = np.where(valid, cleaned, replacement)
    np.fill_diagonal(distances, replacement)

    normalized = np.minimum(distances / safe_scale, 50.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-0.88 * normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + 1e-9 * safe_scale)
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = affinity.sum(axis=1, keepdims=True)
    col_mass = affinity.sum(axis=0, keepdims=True)

    row_preference = affinity / np.maximum(row_mass, eps)
    col_preference = affinity / np.maximum(col_mass, eps)

    mutual = np.sqrt(
        np.maximum(row_preference * col_preference, 0.0)
    )
    mutual[~valid] = 0.0

    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, min(1, n - 1)]

    gap_scale = np.maximum(second - nearest, 0.14 * safe_scale)
    gap_scale = np.maximum(gap_scale, eps)

    local_preference = np.exp(
        -np.clip(
            distances - nearest[:, None],
            0.0,
            None,
        ) / gap_scale[:, None]
    )
    local_preference[~valid] = 0.0

    row_max = np.maximum(
        local_preference.max(axis=1, keepdims=True),
        eps,
    )
    local_preference /= row_max

    regret = np.exp(
        np.clip(
            (second[:, None] - distances) / gap_scale[:, None],
            -18.0,
            18.0,
        )
    )
    regret[~valid] = 0.0

    endpoint_competition = np.sqrt(
        np.maximum(
            local_preference * local_preference.T,
            0.0,
        )
    )
    endpoint_regret = np.sqrt(
        np.maximum(regret * regret.T, 0.0)
    )

    bypass = np.full((n, n), replacement, dtype=float)

    for middle in range(n):
        candidate = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, replacement)

    bypass_gain = np.maximum(bypass - distances, 0.0)
    bypass_scale = max(0.70 * safe_scale, eps)

    irreplaceability = 1.0 - np.exp(
        -bypass_gain / bypass_scale
    )
    irreplaceability[~valid] = 0.0

    forest_edges = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)
    best_edge = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for start in range(n):
        if visited[start]:
            continue

        best_edge[start] = 0.0

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
                other = parent[node]
                forest_edges[node, other] = True
                forest_edges[other, node] = True

            neighbors = np.flatnonzero(
                valid[node] & ~visited
            )

            for other in neighbors:
                if distances[node, other] < best_edge[other]:
                    best_edge[other] = distances[node, other]
                    parent[other] = node

    connectivity = np.full((n, n), 0.88, dtype=float)
    connectivity[forest_edges] = 1.18
    connectivity[~valid] = 0.0

    affinity_max = max(float(np.max(affinity)), eps)
    mutual_max = max(float(np.max(mutual)), eps)

    base = (
        np.maximum(affinity / affinity_max, eps) ** 1.05
        * np.maximum(inverse, eps) ** 0.70
        * np.maximum(mutual / mutual_max, eps) ** 0.90
        * np.maximum(endpoint_competition + 0.10, eps) ** 1.05
        * np.maximum(endpoint_regret + 0.10, eps) ** 0.55
        * np.maximum(irreplaceability + 0.16, eps) ** 0.95
        * connectivity
    )

    base[~valid] = 0.0
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    maximum = float(np.max(base))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    base /= maximum

    balanced = base.copy()

    for _ in range(24):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.64 * base + 0.36 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
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

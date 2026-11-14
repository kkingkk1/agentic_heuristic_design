import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-stability-based edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid = finite & nonnegative
    valid &= valid.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.zeros((n, n), dtype=float)
    symmetric[valid] = 0.5 * (raw[valid] + raw.T[valid])

    positive = symmetric[valid & (symmetric > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    normalized = np.full((n, n), np.inf, dtype=float)
    normalized[valid] = symmetric[valid] / max(scale, eps)
    np.fill_diagonal(normalized, np.inf)

    row_index = np.arange(n, dtype=np.int64)[:, None]
    col_index = np.arange(n, dtype=np.int64)[None, :]
    low = np.minimum(row_index, col_index) + 1
    high = np.maximum(row_index, col_index) + 1

    pair_code = (
        low * np.int64(73856093)
        ^ high * np.int64(19349663)
        ^ np.int64(n + 1) * np.int64(83492791)
    ).astype(np.float64)

    tie_break = 1e-12 * (
        np.mod(np.abs(pair_code), 997.0) / 997.0
    )

    trials = int(np.clip(8 + 2 * np.sqrt(float(n)), 12, 32))
    counts = np.zeros((n, n), dtype=float)

    for trial in range(trials):
        visited = np.zeros(n, dtype=bool)
        best_cost = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=np.int64)

        for step in range(n):
            if step == 0:
                candidate = trial % n
                if visited[candidate]:
                    candidate = int(np.flatnonzero(~visited)[0])
            else:
                unvisited = ~visited
                if np.any(unvisited):
                    masked_cost = np.where(unvisited, best_cost, np.inf)
                    candidate = int(np.argmin(masked_cost))

                    if not np.isfinite(masked_cost[candidate]):
                        candidate = int(np.flatnonzero(unvisited)[0])

            if visited[candidate]:
                remaining = np.flatnonzero(~visited)
                if remaining.size == 0:
                    break
                candidate = int(remaining[0])

            if parent[candidate] >= 0:
                left = candidate
                right = int(parent[candidate])
                counts[left, right] += 1.0
                counts[right, left] += 1.0

            visited[candidate] = True

            phase = (
                pair_code[candidate] * 0.000001
                + (trial + 1) * 1.61803398875
            )
            perturbation = 1.0 + 0.24 * np.sin(phase)

            candidate_cost = normalized[candidate] * perturbation
            candidate_cost = candidate_cost + tie_break[candidate]

            update = (~visited) & (candidate_cost < best_cost)
            best_cost[update] = candidate_cost[update]
            parent[update] = candidate

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_valid = valid[upper_i, upper_j]

    edge_i = upper_i[upper_valid]
    edge_j = upper_j[upper_valid]

    stability = counts[edge_i, edge_j] / float(max(trials, 1))

    distances = symmetric[edge_i, edge_j]
    affinity = np.exp(
        -np.minimum(distances / max(scale, eps), 50.0)
    )
    affinity /= max(float(np.max(affinity)), eps)

    incident = np.zeros(n, dtype=float)
    np.add.at(incident, edge_i, stability)
    np.add.at(incident, edge_j, stability)

    incident_scale = np.maximum(incident, eps)
    endpoint_balance = np.sqrt(
        (
            stability
            + 0.08 / incident_scale[edge_i]
        )
        * (
            stability
            + 0.08 / incident_scale[edge_j]
        )
    )

    scores = (
        0.70 * stability
        + 0.20 * affinity
        + 0.10 * endpoint_balance
    )

    result = np.zeros((n, n), dtype=float)
    result[edge_i, edge_j] = scores
    result[edge_j, edge_i] = scores
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
    )

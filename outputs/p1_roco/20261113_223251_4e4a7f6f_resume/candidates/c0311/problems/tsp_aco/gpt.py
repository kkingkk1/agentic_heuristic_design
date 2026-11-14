import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stable, direction-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    values = raw[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        q25, q75 = np.quantile(positive, [0.25, 0.75])
        mad = float(np.median(np.abs(positive - median)))
        scale = max(
            float(q75 - q25) / 1.349,
            mad / 0.6745,
            0.10 * max(median, 1.0),
            eps,
        )
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(values))
    replacement = maximum + 6.0 * scale

    distances = np.where(valid, raw, replacement)
    np.fill_diagonal(distances, replacement)

    normalized = np.clip(distances / scale, 0.0, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + 0.08 * scale)

    inverse_max = float(np.max(inverse))
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        inverse /= inverse_max

    distance_signal = 0.82 * affinity + 0.18 * inverse
    distance_signal[~valid] = 0.0

    row_mass = distance_signal.sum(axis=1, keepdims=True)
    col_mass = distance_signal.sum(axis=0, keepdims=True)

    row_preference = distance_signal / np.maximum(row_mass, eps)
    col_preference = distance_signal / np.maximum(col_mass, eps)

    mutual = np.sqrt(np.maximum(row_preference * col_preference, 0.0))
    mutual[~valid] = 0.0

    order = np.argsort(distances, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_signal = np.exp(-rank / rank_scale)
    rank_signal[~valid] = 0.0

    sorted_rows = np.sort(np.where(valid, distances, np.inf), axis=1)
    best_row = sorted_rows[:, 0]
    second_row = sorted_rows[:, min(1, n - 1)]
    second_row = np.where(
        np.isfinite(second_row),
        second_row,
        best_row + scale,
    )

    row_gap = np.maximum(second_row - best_row, 0.20 * scale)
    row_regret = np.maximum(second_row[:, None] - distances, 0.0)
    row_regret = 1.0 - np.exp(
        -row_regret / np.maximum(row_gap[:, None], eps)
    )

    sorted_cols = np.sort(np.where(valid, distances, np.inf), axis=0)
    best_col = sorted_cols[0, :]
    second_col = sorted_cols[min(1, n - 1), :]
    second_col = np.where(
        np.isfinite(second_col),
        second_col,
        best_col + scale,
    )

    col_gap = np.maximum(second_col - best_col, 0.20 * scale)
    col_regret = np.maximum(second_col[None, :] - distances, 0.0)
    col_regret = 1.0 - np.exp(
        -col_regret / np.maximum(col_gap[None, :], eps)
    )

    regret = 0.5 * (row_regret + col_regret)
    regret[~valid] = 0.0

    successor_count = valid.sum(axis=1).astype(float)
    predecessor_count = valid.sum(axis=0).astype(float)

    successor_scarcity = np.zeros(n, dtype=float)
    predecessor_scarcity = np.zeros(n, dtype=float)

    active_successors = successor_count > 0.0
    active_predecessors = predecessor_count > 0.0

    successor_scarcity[active_successors] = (
        1.0 / np.sqrt(successor_count[active_successors])
    )
    predecessor_scarcity[active_predecessors] = (
        1.0 / np.sqrt(predecessor_count[active_predecessors])
    )

    if np.any(active_successors):
        successor_scarcity /= max(
            float(np.max(successor_scarcity)),
            eps,
        )

    if np.any(active_predecessors):
        predecessor_scarcity /= max(
            float(np.max(predecessor_scarcity)),
            eps,
        )

    scarcity = 0.5 * (
        successor_scarcity[:, None]
        + predecessor_scarcity[None, :]
    )
    scarcity[~valid] = 0.0

    # Use an undirected projection only to identify a light connectivity bridge prior.
    undirected_valid = np.triu(valid | valid.T, k=1)
    bridge = np.zeros((n, n), dtype=float)

    edge_rows, edge_cols = np.where(undirected_valid)
    if edge_rows.size:
        projected_cost = np.where(
            valid[edge_rows, edge_cols] & valid[edge_cols, edge_rows],
            0.5
            * (
                distances[edge_rows, edge_cols]
                + distances[edge_cols, edge_rows]
            ),
            np.where(
                valid[edge_rows, edge_cols],
                distances[edge_rows, edge_cols],
                distances[edge_cols, edge_rows],
            ),
        )

        edge_order = np.lexsort(
            (edge_cols, edge_rows, projected_cost)
        )

        parent = np.arange(n, dtype=int)
        size = np.ones(n, dtype=int)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for position in edge_order:
            left = int(edge_rows[position])
            right = int(edge_cols[position])

            root_left = find(left)
            root_right = find(right)

            if root_left == root_right:
                continue

            if size[root_left] < size[root_right]:
                root_left, root_right = root_right, root_left

            parent[root_right] = root_left
            size[root_left] += size[root_right]

            if valid[left, right]:
                bridge[left, right] = 1.0
            if valid[right, left]:
                bridge[right, left] = 1.0

    score = (
        0.44 * distance_signal
        + 0.25 * mutual
        + 0.16 * row_regret
        + 0.08 * rank_signal
        + 0.04 * scarcity
        + 0.03 * bridge
    )

    score[~valid] = 0.0

    positive = score[valid]
    if positive.size:
        maximum_score = float(np.max(positive))
        floor = 0.025 * maximum_score
        score[valid] = floor + 0.975 * score[valid]

    score = np.maximum(score, 0.0)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum_score = float(np.max(score))
    if not np.isfinite(maximum_score) or maximum_score <= 0.0:
        result[valid] = 1.0
        return result

    result = score / maximum_score

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

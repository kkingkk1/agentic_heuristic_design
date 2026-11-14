import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic edge priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

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
    if inverse_max > 0.0:
        inverse /= inverse_max

    distance_signal = 0.82 * affinity + 0.18 * inverse
    distance_signal[~valid] = 0.0

    row_mass = distance_signal.sum(axis=1)
    col_mass = distance_signal.sum(axis=0)

    row_preference = distance_signal / np.maximum(row_mass[:, None], eps)
    col_preference = distance_signal / np.maximum(col_mass[None, :], eps)

    mutual = np.sqrt(np.maximum(row_preference * col_preference, 0.0))
    mutual[~valid] = 0.0

    order = np.argsort(distances, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_signal = np.exp(-rank / rank_scale)
    rank_signal[~valid] = 0.0

    sorted_distances = np.sort(distances, axis=1)
    best = sorted_distances[:, 0]
    second_index = min(1, n - 1)
    second = sorted_distances[:, second_index]
    second = np.where(np.isfinite(second), second, best + scale)

    gap_scale = np.maximum(second - best, 0.20 * scale)
    gap = np.maximum(second[:, None] - distances, 0.0)

    row_competition = 1.0 - np.exp(-gap / gap_scale[:, None])
    row_competition[~valid] = 0.0

    # For directed inputs, retain the direction-specific local competition.
    competition = row_competition

    # Build a deterministic undirected projection only for connectivity bridging.
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
        0.47 * distance_signal
        + 0.27 * mutual
        + 0.18 * competition
        + 0.06 * rank_signal
        + 0.02 * bridge
    )
    score[~valid] = 0.0

    positive_scores = score[valid]
    if positive_scores.size:
        maximum_score = float(np.max(positive_scores))
        floor = 0.025 * maximum_score
        score[valid] = floor + 0.975 * score[valid]

    score[~valid] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum_score = float(np.max(score))
    if not np.isfinite(maximum_score) or maximum_score <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    result = score / maximum_score
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

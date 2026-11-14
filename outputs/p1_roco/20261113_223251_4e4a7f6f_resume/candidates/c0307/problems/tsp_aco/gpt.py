import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic, symmetric, non-negative edge priors for TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid | directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)

    both = directed_valid & directed_valid.T
    forward_only = directed_valid & ~directed_valid.T
    reverse_only = directed_valid.T & ~directed_valid

    symmetric[both] = 0.5 * (raw[both] + raw.T[both])
    symmetric[forward_only] = raw[forward_only]
    symmetric[reverse_only] = raw.T[reverse_only]
    np.fill_diagonal(symmetric, np.inf)

    values = symmetric[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        q25, q75 = np.quantile(positive, [0.25, 0.75])
        mad = float(np.median(np.abs(positive - median)))
        scale = max(
            float(q75 - q25) / 1.349,
            mad / 0.6745,
            0.10 * max(median, 1.0),
            np.finfo(float).eps,
        )
    else:
        median = 0.0
        scale = 1.0

    maximum = float(np.max(values))
    replacement = maximum + 6.0 * scale
    distances = np.where(valid, symmetric, replacement)
    np.fill_diagonal(distances, replacement)

    eps = np.finfo(float).eps
    scaled = np.clip(distances / max(scale, eps), 0.0, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-scaled[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        distances[valid] + 0.08 * max(scale, eps)
    )
    inverse_max = float(np.max(inverse))
    if inverse_max > 0.0:
        inverse /= inverse_max

    distance_signal = 0.80 * affinity + 0.20 * inverse
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
    second = sorted_distances[:, min(1, n - 1)]
    second = np.where(np.isfinite(second), second, best + scale)

    gap_scale = np.maximum(second - best, 0.20 * max(scale, eps))
    gap = np.maximum(second[:, None] - distances, 0.0)

    row_competition = 1.0 - np.exp(-gap / gap_scale[:, None])
    row_competition[~valid] = 0.0
    competition = 0.5 * (row_competition + row_competition.T)
    competition[~valid] = 0.0

    # Deterministic minimum-spanning-forest bridge signal.
    bridge = np.zeros((n, n), dtype=float)
    edge_rows, edge_cols = np.where(
        np.triu(valid, k=1)
    )

    if edge_rows.size:
        order_edges = np.lexsort(
            (edge_cols, edge_rows, distances[edge_rows, edge_cols])
        )
        parent = np.arange(n, dtype=int)
        size = np.ones(n, dtype=int)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for index in order_edges:
            left = int(edge_rows[index])
            right = int(edge_cols[index])
            root_left = find(left)
            root_right = find(right)

            if root_left == root_right:
                continue

            if size[root_left] < size[root_right]:
                root_left, root_right = root_right, root_left

            parent[root_right] = root_left
            size[root_left] += size[root_right]
            bridge[left, right] = 1.0
            bridge[right, left] = 1.0

    score = (
        0.43 * distance_signal
        + 0.27 * mutual
        + 0.19 * competition
        + 0.08 * rank_signal
        + 0.03 * bridge
    )
    score[~valid] = 0.0

    positive_scores = score[valid]
    if positive_scores.size:
        score_max = float(np.max(positive_scores))
        floor = 0.025 * score_max
        score[valid] = floor + 0.975 * score[valid]

    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    result = score / maximum
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

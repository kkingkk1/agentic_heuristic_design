import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    values = distances[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    q90 = float(np.quantile(values, 0.90))
    penalty = max(q90, scale) + 12.0 * scale

    clean = np.where(valid, distances, penalty)
    np.fill_diagonal(clean, penalty)

    temperature = max(0.62 * scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 60.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(clean[valid], eps)
    inverse /= max(float(inverse.max()), eps)

    support_mass = affinity.sum(axis=1)
    support = affinity / np.maximum(support_mass[:, None], eps)
    mutual_support = np.sqrt(
        np.maximum(support * support.T, 0.0)
    )
    mutual_support[~valid] = 0.0

    order = np.argsort(clean, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.82 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_scale)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )

    scarcity = 1.0 / np.sqrt(np.maximum(support_mass, eps))
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.65, 1.8)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    nearest = np.min(clean, axis=1)
    second = np.partition(clean, 1, axis=1)[:, 1]
    gap_scale = np.maximum(
        second - nearest,
        0.15 * scale,
    )

    gap = np.where(
        clean <= nearest[:, None] + eps,
        second[:, None] - clean,
        nearest[:, None] - clean,
    )
    exclusivity = np.exp(
        np.clip(gap / gap_scale[:, None], -24.0, 24.0)
    )
    exclusivity[~valid] = 0.0

    mutual_exclusivity = np.sqrt(
        np.maximum(
            exclusivity * exclusivity.T,
            0.0,
        )
    )

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via = (
            clean[:, start:stop, None]
            + clean[None, start:stop, :]
        )
        bypass = np.minimum(
            bypass,
            np.min(via, axis=1),
        )

    np.fill_diagonal(bypass, penalty)

    advantage = np.maximum(bypass - clean, 0.0)
    irreplaceability = 1.0 - np.exp(
        -advantage / max(0.72 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    forest_edges = np.zeros((n, n), dtype=bool)
    visited = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)

    for root in range(n):
        if visited[root]:
            continue

        best[root] = 0.0

        while True:
            candidates = np.where(~visited, best, np.inf)
            node = int(np.argmin(candidates))

            if not np.isfinite(candidates[node]):
                break

            visited[node] = True
            predecessor = parent[node]

            if predecessor >= 0:
                forest_edges[node, predecessor] = True
                forest_edges[predecessor, node] = True

            neighbors = np.flatnonzero(valid[node] & ~visited)

            for neighbor in neighbors:
                if clean[node, neighbor] < best[neighbor]:
                    best[neighbor] = clean[node, neighbor]
                    parent[neighbor] = node

    connectivity = np.full((n, n), 0.88, dtype=float)
    connectivity[forest_edges] = 1.18
    connectivity[~valid] = 0.0

    local_signal = (
        0.42 * mutual_support
        + 0.33 * mutual_rank
        + 0.25 * mutual_exclusivity
    )
    local_signal[~valid] = 0.0

    structural_signal = (
        0.68 * irreplaceability
        + 0.20 * endpoint_scarcity
        + 0.12 * connectivity
    )
    structural_signal[~valid] = 0.0

    score = (
        np.maximum(affinity, eps) ** 0.82
        * np.maximum(inverse, eps) ** 0.58
        * np.maximum(0.30 + local_signal, eps)
        * np.maximum(0.50 + structural_signal, eps)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    balanced = score.copy()
    for _ in range(28):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

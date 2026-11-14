import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)

    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & off_diagonal
    )
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(
        valid,
        0.5 * (distances + distances.T),
        np.inf,
    )
    np.fill_diagonal(safe, np.inf)

    valid_values = safe[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(valid_values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = float(n)

    valid_counts = np.sum(valid, axis=1).astype(float)

    def candidate_graph(k: int):
        k = min(max(int(k), 1), n - 1)
        mask = valid & (ranks < float(k))
        selected = np.where(mask, safe, np.inf)
        selected_order = np.argsort(selected, axis=1, kind="stable")[:, :k]
        selected_distances = np.take_along_axis(
            safe,
            selected_order,
            axis=1,
        )
        selected_valid = np.isfinite(selected_distances)
        counts = np.sum(selected_valid, axis=1).astype(float)
        scale = np.sum(
            np.where(selected_valid, selected_distances, 0.0),
            axis=1,
        ) / np.maximum(counts, 1.0)
        scale = np.where(counts > 0.0, scale, global_scale)
        scale = np.maximum(scale, eps)
        return mask, selected_distances, selected_valid, scale, counts

    short_k = min(2, n - 1)
    medium_k = min(4, n - 1)
    long_k = min(8, n - 1)

    short_graph, _, _, short_scale, _ = candidate_graph(short_k)
    medium_graph, medium_distances, medium_valid, medium_scale, medium_counts = (
        candidate_graph(medium_k)
    )
    long_graph, long_distances, long_valid, long_scale, long_counts = (
        candidate_graph(long_k)
    )

    local_scale = np.cbrt(
        np.maximum(short_scale, eps)
        * np.maximum(medium_scale, eps)
        * np.maximum(long_scale, eps)
    )

    pair_local_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_normalized = safe / np.maximum(pair_local_scale, eps)
    global_normalized = safe / global_scale

    distance_score = np.exp(
        -np.clip(
            0.70 * local_normalized + 0.20 * global_normalized,
            0.0,
            50.0,
        )
    )

    rank_score = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    rank_factor = 0.90 + 0.10 * rank_score

    mutual_short = valid & (ranks < float(short_k)) & (ranks.T < float(short_k))
    mutual_medium = valid & (
        (ranks < float(medium_k))
        & (ranks.T < float(medium_k))
    )
    mutual_long = valid & (
        (ranks < float(long_k))
        & (ranks.T < float(long_k))
    )

    mutual_factor = (
        1.0
        + 0.055 * mutual_short.astype(float)
        + 0.035 * mutual_medium.astype(float)
        + 0.020 * mutual_long.astype(float)
    )

    medium_quality = np.where(
        medium_valid,
        np.exp(
            -np.clip(
                medium_distances
                / np.maximum(medium_scale[:, None], eps),
                0.0,
                50.0,
            )
        ),
        0.0,
    )
    medium_sorted = np.sort(medium_quality, axis=1)
    strongest = medium_sorted[:, -1]
    second_strongest = (
        medium_sorted[:, -2]
        if medium_k >= 2
        else np.zeros(n, dtype=float)
    )

    reserve = second_strongest / np.maximum(strongest, eps)
    reserve = np.clip(reserve, 0.0, 1.0)
    reserve_pair = np.sqrt(reserve[:, None] * reserve[None, :])

    edge_competitiveness = np.sqrt(
        np.exp(
            -np.clip(
                safe / np.maximum(local_scale[:, None], eps),
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                safe / np.maximum(local_scale[None, :], eps),
                0.0,
                50.0,
            )
        )
    )

    combined_graph = np.clip(
        short_graph.astype(float)
        + medium_graph.astype(float)
        + long_graph.astype(float),
        0.0,
        1.0,
    )

    degree = np.sum(combined_graph, axis=1)
    common_neighbors = combined_graph @ combined_graph.T
    union_size = (
        degree[:, None]
        + degree[None, :]
        - common_neighbors
    )

    bridge_signal = 1.0 - common_neighbors / np.maximum(union_size, 1.0)
    bridge_signal = np.clip(bridge_signal, 0.0, 1.0)

    rank_gate = np.exp(
        -np.clip(
            np.minimum(ranks, ranks.T) / max(float(medium_k), 1.0),
            0.0,
            6.0,
        )
    )
    sparse_gate = 1.0 / np.sqrt(
        np.maximum(valid_counts[:, None] + valid_counts[None, :], 2.0)
    )
    bridge_signal *= edge_competitiveness * rank_gate * sparse_gate

    persistent = (
        mutual_short
        & mutual_medium
        & mutual_long
    ).astype(float)

    scores = (
        distance_score
        * rank_factor
        * mutual_factor
    )
    scores[~valid] = 0.0
    scores = np.maximum(scores, eps)

    log_scores = np.log(scores)
    log_scores += 0.045 * reserve_pair * edge_competitiveness
    log_scores += 0.035 * bridge_signal
    log_scores += 0.020 * persistent

    scores = np.exp(np.clip(log_scores, -60.0, 30.0))
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)

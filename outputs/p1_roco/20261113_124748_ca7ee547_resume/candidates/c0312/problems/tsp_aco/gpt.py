import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)

    # An undirected edge is usable only when both directed entries are valid.
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distances = 0.5 * (distances + distances.T)
    safe = np.where(valid, symmetric_distances, np.inf)

    valid_values = safe[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif valid_values.size:
        global_scale = max(float(np.median(valid_values)), 1.0)
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    # Stable ordering makes tied distances deterministic.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values[None, :]
    ranks[~valid] = float(n)

    valid_counts = np.sum(valid, axis=1)
    top_k = min(4, n - 1)

    nearest = np.take_along_axis(safe, order[:, :top_k], axis=1)
    nearest_valid = np.isfinite(nearest)
    local_scale = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    ) / np.maximum(np.sum(nearest_valid, axis=1), 1)

    local_scale = np.where(valid_counts > 0, local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_local_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    local_normalized = safe / np.maximum(pair_local_scale, eps)
    global_normalized = safe / global_scale

    # Local density is dominant; the global term prevents overcommitment
    # caused by an unusually small neighborhood scale.
    distance_score = np.exp(
        -np.clip(0.78 * local_normalized + 0.22 * global_normalized, 0.0, 50.0)
    )

    # Mild bidirectional rank preference.
    rank_score = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    rank_factor = 0.90 + 0.10 * rank_score

    # Reward only mutually strong local edges, with a deliberately small effect.
    mutual_top = valid & (ranks < float(top_k)) & (ranks.T < float(top_k))
    mutual_factor = 1.0 + 0.08 * mutual_top.astype(float)

    # A bounded bridge correction favors edges incident to sparse nodes,
    # but only when the edge is not a poor local choice for either endpoint.
    sparsity = 1.0 / np.sqrt(np.maximum(valid_counts.astype(float), 1.0))
    sparse_pair = 0.5 * (sparsity[:, None] + sparsity[None, :])
    good_rank = np.minimum(
        ranks / max(float(top_k), 1.0),
        ranks.T / max(float(top_k), 1.0),
    )
    bridge_factor = 1.0 + 0.06 * sparse_pair * np.exp(
        -np.clip(good_rank, 0.0, 6.0)
    )

    scores = distance_score * rank_factor * mutual_factor * bridge_factor
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

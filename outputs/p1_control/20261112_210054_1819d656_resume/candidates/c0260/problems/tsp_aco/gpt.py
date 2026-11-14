import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, endpoint-aware edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)) * 1e-10, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n) + 0.5))))

    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, global_scale)

    if n > 2:
        alternative = ordered[:, 1]
    else:
        alternative = ordered[:, 0]
    alternative = np.maximum(alternative, global_scale)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_length = d / (pair_scale + eps)

    short_edge_affinity = np.exp(
        -np.clip(normalized_length, 0.0, 40.0)
    )

    endpoint_i = np.exp(
        -np.clip(d / (local_scale[:, None] + eps), 0.0, 40.0)
    )
    endpoint_j = np.exp(
        -np.clip(d / (local_scale[None, :] + eps), 0.0, 40.0)
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(k, 1))
    rank_i = np.exp(-ranks / rank_scale)
    rank_j = rank_i.T
    mutual_rank = np.sqrt(rank_i * rank_j)

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    normalized_margin = (
        alternative_scale - d
    ) / (
        alternative_scale + d + global_scale
    )
    normalized_margin = np.clip(normalized_margin, 0.0, 1.0)

    rank_gate = np.sqrt(
        np.clip(mutual_rank / (mutual_rank + 0.35), 0.0, 1.0)
    )
    margin_bonus = 1.0 + 0.10 * normalized_margin * rank_gate

    score = (
        short_edge_affinity
        * endpoint_compatibility
        * mutual_rank
        * margin_bonus
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

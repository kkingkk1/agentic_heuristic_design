import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & off_diagonal

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
    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    scale_floor = max(float(np.median(positive)) * 1e-10, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    nearest_one = nearest[:, 0]
    nearest_two = nearest[:, min(1, k - 1)]
    if n > 2:
        nearest_two = np.minimum(
            nearest_two,
            np.partition(d, kth=1, axis=1)[:, 1],
        )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = d / (pair_scale + eps)
    base_affinity = np.exp(
        -np.clip(normalized_distance, 0.0, 40.0)
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_decay = max(1.0, 0.5 * (k + 1))
    rank_affinity = np.exp(
        -0.5 * (ranks / rank_decay)
    )
    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)

    relative_saving_i = np.clip(
        (nearest_two[:, None] - d) / (nearest_two[:, None] + eps),
        0.0,
        1.0,
    )
    relative_saving_j = np.clip(
        (nearest_two[None, :] - d) / (nearest_two[None, :] + eps),
        0.0,
        1.0,
    )
    replacement_saving = np.sqrt(
        relative_saving_i * relative_saving_j
    )

    scale_consistency = np.exp(
        -np.abs(
            np.log(
                (local_scale[:, None] + eps)
                / (local_scale[None, :] + eps)
            )
        )
    )

    local_candidate_gate = 0.65 + 0.35 * mutual_rank
    margin_bonus = 1.0 + 0.20 * replacement_saving * local_candidate_gate

    score = (
        base_affinity
        * mutual_rank
        * (0.75 + 0.25 * scale_consistency)
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
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

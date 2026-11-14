import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, normalized edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    positive = distances[symmetric_valid & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    repaired = np.where(valid, distances, fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 0.0)

    work = repaired.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.rint(np.sqrt(n)))))

    neighborhood = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(neighborhood.mean(axis=1), eps)
    substitute = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = repaired / (pair_scale + eps)
    distance_affinity = np.exp(
        -np.clip(relative_distance ** 0.96, 0.0, 50.0)
    )

    row_ratio_i = repaired / (local_scale[:, None] + eps)
    row_ratio_j = repaired / (local_scale[None, :] + eps)

    endpoint_competition = np.sqrt(
        (1.0 / (1.0 + row_ratio_i))
        * (1.0 / (1.0 + row_ratio_j))
    )
    competition_factor = 0.84 + 0.16 * endpoint_competition

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / np.sqrt(1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    candidate_pressure = 0.5 * (
        ranks / max(float(k), 1.0)
        + ranks.T / max(float(k), 1.0)
    )
    candidate_gate = 0.86 + 0.14 * np.exp(
        -np.clip(candidate_pressure, 0.0, 8.0)
    )

    substitute_scale = np.sqrt(
        substitute[:, None] * substitute[None, :]
    )
    margin = substitute_scale / (
        repaired + substitute_scale + eps
    )
    margin_factor = 0.95 + 0.05 * np.clip(margin, 0.0, 1.0)

    local_affinity = np.exp(
        -np.clip(work / (local_scale[:, None] + eps), 0.0, 12.0)
    )
    np.fill_diagonal(local_affinity, 0.0)

    support = local_affinity @ local_affinity.T
    support_norm = np.sqrt(
        np.sum(local_affinity ** 2, axis=1)[:, None]
        * np.sum(local_affinity ** 2, axis=1)[None, :]
    )
    common_support = np.divide(
        support,
        support_norm + eps,
        out=np.zeros_like(support),
        where=np.isfinite(support_norm),
    )
    support_factor = 0.97 + 0.03 * np.clip(common_support, 0.0, 1.0)

    score = (
        distance_affinity
        * mutual_rank
        * competition_factor
        * candidate_gate
        * margin_factor
        * support_factor
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

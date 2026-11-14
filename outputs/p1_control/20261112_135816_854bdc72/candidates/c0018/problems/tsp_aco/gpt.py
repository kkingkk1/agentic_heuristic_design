import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    proximity = 1.0 / (1.0 + d / scale)
    proximity[~finite] = 0.0
    np.fill_diagonal(proximity, 0.0)

    row_order = np.argsort(d, axis=1)
    ranks = np.empty_like(row_order, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    ranks = np.maximum(ranks, 1.0)

    rank_score = 1.0 / np.sqrt(ranks)
    rank_score[~finite] = 0.0
    np.fill_diagonal(rank_score, 0.0)

    mutual_score = np.sqrt(rank_score * rank_score.T)

    alternate = np.minimum(d[:, :, None] + d[None, :, :], np.inf)
    detour = np.min(alternate, axis=2)
    detour = np.minimum(detour, detour.T)
    detour_advantage = np.maximum(detour - d, 0.0)
    detour_advantage[~finite] = 0.0

    advantage_scale = np.median(detour_advantage[detour_advantage > 0])
    if not np.isfinite(advantage_scale) or advantage_scale <= 0:
        advantage_scale = scale

    exclusivity = detour_advantage / (detour_advantage + advantage_scale)
    exclusivity[~finite] = 0.0
    np.fill_diagonal(exclusivity, 0.0)

    score = (
        0.45 * proximity
        + 0.30 * mutual_score
        + 0.25 * exclusivity
    )
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    score = score / row_max

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)

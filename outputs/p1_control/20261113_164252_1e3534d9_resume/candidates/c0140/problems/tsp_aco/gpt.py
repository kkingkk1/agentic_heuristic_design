import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return sparse, diverse, and regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    safe = np.where(valid, d, np.inf)
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    row_rank[rows, row_order] = np.arange(n, dtype=float)[None, :]
    col_rank[col_order, cols] = np.arange(n, dtype=float)[:, None]

    usable_count = max(1, n - 1)
    candidate_k = min(4, usable_count)

    row_min = np.min(safe, axis=1)
    row_second = np.partition(safe, kth=min(1, n - 1), axis=1)[
        :, min(1, n - 1)
    ]
    col_min = np.min(safe, axis=0)
    col_second = np.partition(safe, kth=min(1, n - 1), axis=0)[
        min(1, n - 1), :
    ]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    def robust_scale(values: np.ndarray) -> float:
        positive_values = values[values > 0.0]
        if positive_values.size == 0:
            return global_scale
        return max(float(np.median(positive_values)), eps)

    row_gap_scale = robust_scale(row_gap)
    col_gap_scale = robust_scale(col_gap)

    # Mild suppression of long edges using robust row-relative distances.
    row_excess = np.maximum(d - row_min[:, None], 0.0)
    row_scale = np.maximum(
        np.median(
            np.where(valid, row_excess, np.nan),
            axis=1,
        ),
        global_scale * 0.25,
    )
    row_scale = np.where(np.isfinite(row_scale), row_scale, global_scale)
    distance_score = np.exp(
        -np.clip(row_excess / row_scale[:, None], 0.0, 30.0)
    )
    distance_score[~valid] = 0.0

    # Rank-based nearest-neighbor signal.
    nn_score = np.exp(-row_rank / 1.15)
    nn_score[~valid] = 0.0

    # Strong bonus only when an edge is locally preferred in both directions.
    mutual_nn = ((row_rank == 0.0) & (col_rank == 0.0) & valid).astype(float)

    # Selective row-column competition within a small candidate set.
    row_candidate = np.where(
        row_rank < candidate_k,
        (candidate_k - row_rank) / candidate_k,
        0.0,
    )
    col_candidate = np.where(
        col_rank < candidate_k,
        (candidate_k - col_rank) / candidate_k,
        0.0,
    )
    competition = np.sqrt(row_candidate * col_candidate)
    competition[~valid] = 0.0

    # Regret is used only when excluding the best edge creates a clear gap.
    row_gap_signal = row_gap[:, None] / (
        row_gap[:, None] + row_gap_scale
    )
    col_gap_signal = col_gap[None, :] / (
        col_gap[None, :] + col_gap_scale
    )
    regret_gate = np.sqrt(row_gap_signal * col_gap_signal)

    row_regret = np.exp(
        -np.clip(row_excess / row_gap_scale, 0.0, 30.0)
    )
    col_excess = np.maximum(d - col_min[None, :], 0.0)
    col_regret = np.exp(
        -np.clip(col_excess / col_gap_scale, 0.0, 30.0)
    )
    regret_score = (
        np.sqrt(row_regret * col_regret)
        * regret_gate
        * (0.5 + 0.5 * competition)
    )
    regret_score[~valid] = 0.0

    # Low-weight rank tail preserves candidate diversity beyond nearest neighbors.
    diversity_score = np.exp(-row_rank / 3.0)
    diversity_score[~valid] = 0.0

    def normalize(feature: np.ndarray) -> np.ndarray:
        feature = np.where(valid, feature, 0.0)
        maximum = float(np.max(feature))
        return feature / maximum if maximum > 0.0 else feature

    distance_score = normalize(distance_score)
    nn_score = normalize(nn_score)
    mutual_nn = normalize(mutual_nn)
    competition = normalize(competition)
    regret_score = normalize(regret_score)
    diversity_score = normalize(diversity_score)

    score = (
        0.27 * distance_score
        + 0.25 * nn_score
        + 0.18 * mutual_nn
        + 0.12 * competition
        + 0.10 * regret_score
        + 0.08 * diversity_score
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~np.eye(n, dtype=bool)
    )

    if not np.any(valid):
        return scores

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.max(finite_values)), 1.0)
    )
    scale = max(scale, np.finfo(float).eps)

    penalty = max(float(np.max(finite_values)), scale) * 4.0 + scale
    work = np.where(valid, distances, penalty)

    row_order = np.argsort(work, axis=1, kind="stable")
    col_order = np.argsort(work, axis=0, kind="stable")

    row_ranks = np.empty((n, n), dtype=float)
    row_ranks[np.arange(n)[:, None], row_order] = np.arange(n)

    col_ranks = np.empty((n, n), dtype=float)
    col_ranks[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    sorted_rows = np.sort(work, axis=1)
    sorted_cols = np.sort(work, axis=0)

    row_best = sorted_rows[:, 0]
    row_second = sorted_rows[:, 1]
    col_best = sorted_cols[0, :]
    col_second = sorted_cols[1, :]

    eps = np.finfo(float).eps
    row_scale = np.maximum(0.5 * (row_best + row_second), eps)
    col_scale = np.maximum(0.5 * (col_best + col_second), eps)

    row_relative = work / row_scale[:, None]
    col_relative = work / col_scale[None, :]

    distance_term = np.exp(
        -np.clip(0.5 * (row_relative + col_relative), 0.0, 60.0)
    )

    rank_term = 1.0 / np.sqrt(
        (1.0 + row_ranks) * (1.0 + col_ranks)
    )

    row_access = np.exp(-np.clip(row_relative, 0.0, 60.0))
    col_access = np.exp(-np.clip(col_relative, 0.0, 60.0))
    accessibility = np.sqrt(row_access * col_access)

    row_regret = np.clip(
        (row_second - row_best) / np.maximum(row_second, eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second - col_best) / np.maximum(col_second, eps),
        0.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.30 * (
        row_regret[:, None] + col_regret[None, :]
    )

    nearest_out = row_order[:, 0]
    nearest_in = col_order[0, :]
    mutual_bonus = np.where(
        (nearest_out[:, None] == np.arange(n)[None, :])
        & (nearest_in[None, :] == np.arange(n)[:, None]),
        1.20,
        1.0,
    )

    # Candidate support estimates how strongly each node favors this edge.
    support = 0.5 * (
        np.exp(-np.clip(row_relative, 0.0, 60.0))
        + np.exp(-np.clip(col_relative, 0.0, 60.0))
    )

    # Favor edges incident to nodes with scarce good alternatives.
    row_opportunity = np.sum(
        np.exp(-np.clip(row_relative, 0.0, 12.0)) * valid,
        axis=1,
    )
    col_opportunity = np.sum(
        np.exp(-np.clip(col_relative, 0.0, 12.0)) * valid,
        axis=0,
    )
    row_scarcity = 1.0 / np.sqrt(np.maximum(row_opportunity, 1.0))
    col_scarcity = 1.0 / np.sqrt(np.maximum(col_opportunity, 1.0))
    coverage_bonus = 1.0 + 0.16 * (
        row_scarcity[:, None] + col_scarcity[None, :]
    )

    # Bridge risk is high when neither endpoint has much local support.
    # Keep the penalty weak so necessary inter-cluster edges survive.
    bridge_risk = np.clip(1.0 - support, 0.0, 1.0)
    bridge_penalty = 1.0 - 0.12 * bridge_risk

    # Flat local neighborhoods act as an intrinsic stagnation proxy.
    # Increase exploration only where several candidates are nearly tied.
    row_flatness = np.clip(
        1.0 - np.abs(row_second - row_best)
        / np.maximum(row_second + row_best, eps),
        0.0,
        1.0,
    )
    col_flatness = np.clip(
        1.0 - np.abs(col_second - col_best)
        / np.maximum(col_second + col_best, eps),
        0.0,
        1.0,
    )
    exploration_bonus = 1.0 + 0.10 * (
        row_flatness[:, None] + col_flatness[None, :]
    )

    scores = (
        distance_term
        * (0.48 + 0.27 * rank_term + 0.25 * accessibility)
        * regret_bonus
        * mutual_bonus
        * coverage_bonus
        * bridge_penalty
        * exploration_bonus
    )

    scores[~valid] = 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    np.fill_diagonal(scores, 0.0)
    return scores

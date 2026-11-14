import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    # Retain only edges valid from both endpoints so the output is symmetric.
    valid &= valid.T
    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)
    finite_values = work[np.isfinite(work)]
    scale = np.median(finite_values) if finite_values.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    eps = np.finfo(float).eps

    # Robust inverse-distance preference.
    normalized_distance = work / max(scale, eps)
    inverse = np.divide(
        1.0,
        normalized_distance + 0.03,
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )

    # Rank-based local proximity.
    ranks = np.full((n, n), n + 1.0, dtype=float)
    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size:
            ordered = neighbors[np.argsort(work[i, neighbors], kind="stable")]
            ranks[i, ordered] = np.arange(1, ordered.size + 1)

    rank_score = np.zeros_like(d, dtype=float)
    rank_score[valid] = 1.0 / np.sqrt(ranks[valid])

    # Endpoint neighborhood scales and regret.
    nearest = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)

    for i in range(n):
        values = np.sort(work[i, valid[i]])
        if values.size:
            nearest[i] = values[0]
        if values.size >= 2:
            second[i] = values[1]
        elif values.size == 1:
            second[i] = values[0]

    nearest = np.where(np.isfinite(nearest), nearest, scale)
    second = np.where(np.isfinite(second), second, nearest)

    node_regret = np.divide(
        np.maximum(second - nearest, 0.0),
        np.maximum(second, eps),
    )
    node_regret = np.clip(node_regret, 0.0, 1.0)

    endpoint_scale = np.sqrt(
        np.maximum(nearest[:, None], eps)
        * np.maximum(nearest[None, :], eps)
    )

    relative_attractiveness = np.exp(
        -np.clip(
            np.divide(
                work,
                endpoint_scale,
                out=np.full_like(d, np.inf),
                where=endpoint_scale > 0,
            ),
            0.0,
            60.0,
        )
    )

    # Mutual local preference.
    mutual = np.divide(
        nearest[:, None] * nearest[None, :],
        np.maximum(work * work, eps),
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )
    mutual = np.sqrt(np.clip(mutual, 0.0, 1.0))

    # Mild candidate-list suppression for unusually long edges.
    row_median = np.full(n, scale, dtype=float)
    for i in range(n):
        values = work[i, valid[i]]
        if values.size:
            row_median[i] = np.median(values)

    candidate_gate = np.divide(
        1.0,
        1.0
        + np.maximum(
            work - np.minimum(row_median[:, None], row_median[None, :]),
            0.0,
        )
        / max(scale, eps),
        out=np.zeros_like(d, dtype=float),
        where=valid,
    )

    regret_bonus = 1.0 + 0.35 * (
        node_regret[:, None] + node_regret[None, :]
    )

    score = (
        0.42 * inverse
        + 0.23 * rank_score
        + 0.20 * mutual
        + 0.15 * relative_attractiveness
    )
    score *= regret_bonus
    score *= 0.65 + 0.35 * candidate_gate
    score[~valid] = 0.0

    # Row normalization limits domination by high-degree hubs.
    row_totals = score.sum(axis=1, keepdims=True)
    row_normalized = np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0,
    )

    score = 0.5 * (row_normalized + row_normalized.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

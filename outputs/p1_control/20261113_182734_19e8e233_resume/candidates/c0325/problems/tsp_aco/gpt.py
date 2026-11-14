import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank- and triangle-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = np.where(valid, d / scale, np.inf)

    normalized = np.where(
        valid,
        np.minimum(normalized, 1.0e6),
        np.inf,
    )

    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    row_order = np.argsort(normalized, axis=1, kind="stable")
    col_order = np.argsort(normalized, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_positions = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    col_positions = np.broadcast_to(
        np.arange(n, dtype=float)[:, None],
        (n, n),
    )

    np.put_along_axis(row_rank, row_order, row_positions, axis=1)
    np.put_along_axis(col_rank, col_order, col_positions, axis=0)

    row_denominator = np.maximum(row_count[:, None] - 1.0, 1.0)
    col_denominator = np.maximum(col_count[None, :] - 1.0, 1.0)

    row_fraction = row_rank / row_denominator
    col_fraction = col_rank / col_denominator

    row_fraction[~valid] = 1.0
    col_fraction[~valid] = 1.0

    row_kernel = np.exp(-3.2 * row_fraction)
    col_kernel = np.exp(-2.6 * col_fraction)

    row_support = np.sqrt(
        np.divide(
            1.0,
            np.maximum(row_count, 1.0),
        )
    )
    col_support = np.sqrt(
        np.divide(
            1.0,
            np.maximum(col_count, 1.0),
        )
    )

    support_bonus = (
        0.55
        + 0.45
        * np.minimum(
            row_support[:, None] + col_support[None, :],
            1.0,
        )
    )

    with np.errstate(over="ignore", invalid="ignore"):
        two_hop = np.min(
            normalized[:, :, None] + normalized[None, :, :],
            axis=1,
        )

    direct_advantage = np.zeros((n, n), dtype=float)
    finite_two_hop = np.isfinite(two_hop) & valid

    with np.errstate(over="ignore", invalid="ignore"):
        np.subtract(
            two_hop,
            normalized,
            out=direct_advantage,
            where=finite_two_hop,
        )

    direct_advantage = np.clip(direct_advantage, -12.0, 12.0)

    with np.errstate(over="ignore", invalid="ignore"):
        bridge_bonus = 0.72 + 0.68 / (
            1.0 + np.exp(-1.8 * direct_advantage)
        )

    bridge_bonus[~finite_two_hop] = 0.72
    bridge_bonus[~valid] = 0.0

    score = (
        row_kernel
        * col_kernel
        * support_bonus
        * bridge_bonus
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

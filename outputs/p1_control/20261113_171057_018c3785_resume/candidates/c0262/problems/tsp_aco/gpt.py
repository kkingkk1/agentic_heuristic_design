import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed TSP edge priors from rank reciprocity and congestion control."""
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
    safe = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
    scale = max(scale, eps)

    row_span = np.max(
        np.where(valid, d, -np.inf),
        axis=1,
    ) - row_min
    col_span = np.max(
        np.where(valid, d, -np.inf),
        axis=0,
    ) - col_min

    row_span = np.where(
        (row_count > 0) & np.isfinite(row_span) & (row_span > eps),
        row_span,
        scale,
    )
    col_span = np.where(
        (col_count > 0) & np.isfinite(col_span) & (col_span > eps),
        col_span,
        scale,
    )

    row_relative = np.zeros_like(d)
    col_relative = np.zeros_like(d)

    np.divide(
        d - row_min[:, None],
        row_span[:, None],
        out=row_relative,
        where=valid,
    )
    np.divide(
        d - col_min[None, :],
        col_span[None, :],
        out=col_relative,
        where=valid,
    )

    row_order = np.argsort(
        np.where(valid, d, np.inf),
        axis=1,
        kind="stable",
    )
    col_order = np.argsort(
        np.where(valid, d, np.inf),
        axis=0,
        kind="stable",
    )

    row_rank = np.empty_like(row_order, dtype=float)
    col_rank = np.empty_like(col_order, dtype=float)

    positions = np.arange(n, dtype=float)
    for i in range(n):
        row_rank[i, row_order[i]] = positions
        col_rank[col_order[:, i], i] = positions

    row_denominator = np.maximum(row_count[:, None] - 1.0, 1.0)
    col_denominator = np.maximum(col_count[None, :] - 1.0, 1.0)

    row_rank = row_rank / row_denominator
    col_rank = col_rank / col_denominator

    rank_priority = np.exp(
        -2.4 * np.sqrt(np.maximum(row_rank, 0.0))
        -2.0 * np.sqrt(np.maximum(col_rank, 0.0))
    )

    cost_priority = np.exp(
        -1.35 * np.sqrt(np.maximum(row_relative, 0.0))
        -0.85 * np.sqrt(np.maximum(col_relative, 0.0))
    )

    score = rank_priority * cost_priority
    score[~valid] = 0.0

    incoming = score.sum(axis=0)
    positive_incoming = incoming[incoming > 0.0]
    reference_incoming = (
        float(np.median(positive_incoming))
        if positive_incoming.size
        else 1.0
    )
    reference_incoming = max(reference_incoming, eps)

    congestion = np.sqrt(
        reference_incoming
        / np.maximum(incoming, reference_incoming)
    )
    score *= congestion[None, :]

    row_mass = score.sum(axis=1, keepdims=True)
    np.divide(
        score,
        row_mass,
        out=result,
        where=row_mass > 0.0,
    )

    result *= np.sqrt(np.maximum(score, 0.0) + eps)
    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(result.max())
    if maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)

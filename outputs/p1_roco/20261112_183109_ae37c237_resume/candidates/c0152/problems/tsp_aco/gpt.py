import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directional one-step-lookahead priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = distances[valid & (distances > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    safe_distance = np.where(valid, distances, np.inf)

    row_counts = valid.sum(axis=1)
    col_counts = valid.sum(axis=0)

    row_sum = np.sum(
        np.where(valid, np.log1p(distances / global_scale), 0.0),
        axis=1,
    )
    col_sum = np.sum(
        np.where(valid, np.log1p(distances / global_scale), 0.0),
        axis=0,
    )

    row_mean = np.divide(
        row_sum,
        np.maximum(row_counts, 1),
        out=np.zeros(n, dtype=float),
        where=row_counts > 0,
    )
    col_mean = np.divide(
        col_sum,
        np.maximum(col_counts, 1),
        out=np.zeros(n, dtype=float),
        where=col_counts > 0,
    )

    successor_cost = np.min(
        np.where(valid, np.log1p(safe_distance / global_scale), np.inf),
        axis=1,
    )
    successor_cost = np.where(
        np.isfinite(successor_cost),
        successor_cost,
        np.max(row_mean, initial=1.0),
    )

    source_scale = np.maximum(row_mean, eps)
    target_scale = np.maximum(col_mean, eps)

    edge_cost = np.log1p(
        np.divide(
            np.where(valid, distances, 0.0),
            global_scale,
        )
    )

    lookahead = successor_cost[None, :]
    source_mismatch = np.divide(
        edge_cost,
        source_scale[:, None],
        out=np.zeros_like(edge_cost),
        where=source_scale[:, None] > 0.0,
    )
    target_mismatch = np.divide(
        lookahead,
        target_scale[None, :],
        out=np.zeros_like(edge_cost),
        where=target_scale[None, :] > 0.0,
    )

    destination_access = np.divide(
        np.maximum(col_counts, 1),
        np.maximum(row_counts, 1),
    )[None, :]

    effective_cost = (
        0.62 * edge_cost
        + 0.23 * lookahead
        + 0.10 * source_mismatch
        + 0.05 * target_mismatch
    )

    temperature = 0.85 + 0.15 * np.clip(destination_access, 0.5, 2.0)
    score = np.exp(-np.clip(temperature * effective_cost, 0.0, 60.0))

    score *= 0.85 + 0.15 * np.exp(
        -np.clip(np.abs(source_mismatch - target_mismatch), 0.0, 60.0)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )

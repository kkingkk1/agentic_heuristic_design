import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, locality-aware prior for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    # Preserve only edges usable in both directions.
    symmetric_valid = valid & valid.T
    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    working = np.where(symmetric_valid, 0.5 * (distances + distances.T), np.inf)
    np.fill_diagonal(working, np.inf)

    positive = working[np.isfinite(working) & (working > 0.0)]
    finite = working[np.isfinite(working)]
    eps = np.finfo(float).eps

    if positive.size:
        global_scale = float(np.median(positive))
    elif finite.size:
        global_scale = max(float(np.median(finite)), 1.0)
    else:
        return np.zeros((n, n), dtype=float)
    global_scale = max(global_scale, eps)

    # Obtain a robust local scale from the nearest up to three neighbors.
    usable = min(3, n - 1)
    nearest_values = np.partition(working, kth=usable - 1, axis=1)[:, :usable]
    local_scale = np.mean(
        np.where(np.isfinite(nearest_values), nearest_values, global_scale),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    # Bidirectional shortness with adaptive temperature.
    relative = np.divide(
        working,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=symmetric_valid,
    )
    shortness = np.exp(-np.clip(relative, 0.0, 50.0))

    # Reciprocal rank scarcity without constructing a full sort order.
    shorter = (
        np.sum(
            working[:, :, None] > working[:, None, :],
            axis=2,
        )
        if False
        else None
    )
    ranks = np.sum(
        working[:, :, None] > working[:, None, :],
        axis=2,
        dtype=np.int32,
    )
    # The diagonal and invalid entries are excluded from the comparison.
    rank_score = 1.0 / np.sqrt(
        (1.0 + ranks.astype(float))
        * (1.0 + ranks.T.astype(float))
    )

    # Mutual nearest-neighbor reinforcement.
    nearest = np.argmin(working, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    mutual_rows = rows[
        np.isfinite(working[rows, nearest])
        & (nearest[nearest] == rows)
    ]
    mutual[mutual_rows, nearest[mutual_rows]] = 1.0
    mutual[nearest[mutual_rows], mutual_rows] = 1.0

    # Bridge pressure for locally isolated endpoints.
    isolation = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) / (working + global_scale)
    isolation = np.nan_to_num(
        isolation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    isolation = np.clip(isolation, 0.0, 1.0)

    # A small bridge component retains useful connections in sparse regions.
    bridge = np.exp(
        -np.clip(working / (local_scale[:, None] + local_scale[None, :]), 0.0, 50.0)
    )

    score = (
        shortness
        * (0.50 + 0.50 * rank_score)
        * (1.0 + 0.30 * mutual)
        * (1.0 + 0.12 * isolation)
        + 0.08 * bridge * isolation
    )

    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

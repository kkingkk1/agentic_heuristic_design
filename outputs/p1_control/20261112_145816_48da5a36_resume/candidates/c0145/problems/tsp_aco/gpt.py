import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d > 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)), eps)

    safe_d = np.where(valid, d, np.inf)

    # Use the second-nearest-neighbor scale when available.
    kth = min(1, n - 2)
    local_scale = np.partition(safe_d, kth, axis=1)[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    normalized_distance = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = np.nan_to_num(
        normalized_distance,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )

    # Reciprocal-distance signal with a soft tail.
    reciprocal_score = 1.0 / (1.0 + normalized_distance)

    # Rank signal from both endpoints.
    rank_order = np.argsort(safe_d, axis=1, kind="stable")
    rank_position = np.empty_like(rank_order, dtype=float)
    rank_position[
        np.arange(n)[:, None],
        rank_order,
    ] = np.arange(n, dtype=float)

    rank_fraction = rank_position / max(n - 1, 1)
    endpoint_rank = np.exp(
        -1.8 * (rank_fraction + rank_fraction.T)
    )

    # Reward edges that are mutually preferred by both endpoints.
    mutual_rank = np.exp(
        -2.4 * np.maximum(rank_fraction, rank_fraction.T)
    )

    # Prefer edges whose endpoints have compatible local distance scales.
    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )
    scale_compatibility = np.exp(-0.8 * scale_ratio)

    # Estimate local crowding, then apply only a bounded penalty.
    density_kernel = np.exp(
        -np.minimum(
            d / local_scale[:, None],
            40.0,
        )
    )
    density_kernel *= valid
    density = np.sum(density_kernel, axis=1)

    crowding_factor = 1.0 / np.sqrt(
        (1.0 + density[:, None])
        * (1.0 + density[None, :])
    )

    # Keep a floor so cross-density edges are not eliminated.
    crowding_factor = 0.60 + 0.40 * crowding_factor

    score = (
        0.36 * reciprocal_score
        + 0.28 * endpoint_rank
        + 0.18 * mutual_rank
        + 0.18 * scale_compatibility
    )
    score *= crowding_factor

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(
        float,
        copy=False,
    )

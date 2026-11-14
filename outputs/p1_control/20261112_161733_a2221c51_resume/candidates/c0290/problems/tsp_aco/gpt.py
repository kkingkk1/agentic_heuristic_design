import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    with np.errstate(invalid="ignore"):
        d = 0.5 * (d + d.T)

    eps = np.finfo(float).eps
    positive_mask = valid & (d > 0.0)

    positive_values = d[positive_mask]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), np.inf)

    positive_work = np.where(positive_mask, d, np.inf)
    if n > 1:
        nearest_two = np.partition(
            positive_work,
            kth=1,
            axis=1,
        )[:, :2]
        first_positive = nearest_two[:, 0]
        second_positive = nearest_two[:, 1]
    else:
        first_positive = positive_work[:, 0]
        second_positive = np.full(n, np.inf)

    first_finite = np.min(work, axis=1)
    local_scale = np.where(
        np.isfinite(second_positive),
        second_positive,
        np.where(
            np.isfinite(first_positive),
            first_positive,
            np.where(
                np.isfinite(first_finite),
                first_finite,
                global_scale,
            ),
        ),
    )
    local_scale = np.maximum(local_scale, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_length = d / np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        )
    normalized_length = np.where(valid, normalized_length, np.inf)
    normalized_length = np.minimum(normalized_length, 1e6)

    order = np.argsort(work, axis=1, kind="stable")
    rank_by_column = np.empty((n, n), dtype=float)
    np.put_along_axis(
        rank_by_column,
        order,
        np.arange(n, dtype=float)[None, :],
        axis=1,
    )

    valid_count = valid.sum(axis=1).astype(float)
    rank_denominator = np.maximum(valid_count - 1.0, 1.0)
    ranks = rank_by_column / rank_denominator[:, None]

    bilateral_rank_quality = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank_quality = np.maximum(bilateral_rank_quality, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        reciprocal = global_scale / np.maximum(d, eps)
    reciprocal = np.where(valid, np.minimum(reciprocal, 1e12), 0.0)

    nearest = np.argmin(work, axis=1)
    node_ids = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == node_ids[None, :])
        & (nearest[None, :] == node_ids[:, None])
        & valid
    ).astype(float)

    density_radius = 1.5 * local_scale
    local_density = np.mean(
        valid & (work <= density_radius[:, None]),
        axis=1,
    )

    excess_density = np.maximum(local_density - 0.35, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(normalized_length, 3.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        scale_mismatch = np.abs(
            np.log(
                np.maximum(local_scale[:, None], eps)
                / np.maximum(local_scale[None, :], eps)
            )
        )
    scale_penalty = np.minimum(scale_mismatch, 12.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_score = (
            0.90 * np.log(np.maximum(reciprocal, eps))
            + 0.85 * np.log(bilateral_rank_quality)
            - 0.85 * normalized_length
            + 0.28 * mutual_nearest
            - 0.10 * scale_penalty
            - 0.08 * density_penalty
        )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

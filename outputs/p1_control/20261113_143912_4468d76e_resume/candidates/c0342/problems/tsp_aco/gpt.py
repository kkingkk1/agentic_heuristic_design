import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric = np.zeros_like(d, dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric[valid & (symmetric > 0.0)]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    effective = np.where(
        valid,
        np.maximum(symmetric, eps * global_scale),
        np.inf,
    )

    degree = valid.sum(axis=1).astype(float)
    degree_fraction = degree / max(n - 1, 1)

    # Use the second available neighbor as a robust local scale.
    # For n=2, the only neighbor is used instead.
    kth = min(1, n - 2)
    local_scale = np.partition(effective, kth=kth, axis=1)[:, kth]

    finite_local = np.isfinite(local_scale)
    if np.any(~finite_local):
        row_values = np.where(valid, effective, np.nan)
        row_medians = np.nanmedian(row_values, axis=1)
        row_medians = np.where(
            np.isfinite(row_medians),
            row_medians,
            global_scale,
        )
        local_scale = np.where(
            finite_local,
            local_scale,
            row_medians,
        )

    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    relative_length = effective / pair_scale
    relative_length = np.minimum(relative_length, 1e6)

    global_length = effective / global_scale
    global_length = np.minimum(global_length, 1e6)

    # Stable bilateral rank among valid neighbors.
    order = np.argsort(effective, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    rank_denominator = np.maximum(degree - 1.0, 1.0)
    normalized_rank = rank / rank_denominator[:, None]
    normalized_rank = np.where(valid, normalized_rank, 1.0)

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - normalized_rank, eps)
        * np.maximum(1.0 - normalized_rank.T, eps)
    )

    # Wider tolerance for sparse endpoints.
    sparsity = 1.0 / np.sqrt(
        np.maximum(degree_fraction[:, None], eps)
        * np.maximum(degree_fraction[None, :], eps)
    )
    sparsity = np.minimum(sparsity, 8.0)

    temperature = 0.85 + 0.45 * np.minimum(sparsity - 1.0, 3.0)
    local_penalty = np.minimum(relative_length / temperature, 18.0)

    # Reward edges that are plausible from both endpoint neighborhoods.
    candidate_radius = 1.8 * np.maximum(
        local_scale[:, None],
        local_scale[None, :],
    )
    reciprocal_candidate = (
        valid
        & (symmetric <= candidate_radius)
        & (symmetric <= candidate_radius.T)
    )

    # Penalize large local-scale mismatch without eliminating such edges.
    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_mismatch = np.minimum(scale_mismatch, 12.0)

    log_score = (
        1.15 * np.log(
            np.maximum(global_scale / np.maximum(effective, eps), eps)
        )
        + 1.20 * np.log(bilateral_rank)
        - 0.82 * local_penalty
        - 0.20 * np.minimum(global_length, 24.0)
        - 0.10 * scale_mismatch
        + 0.38 * reciprocal_candidate.astype(float)
        + 0.10 * np.log(sparsity)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score = np.where(valid, score, 0.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

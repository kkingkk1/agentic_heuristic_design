import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d > 0.0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros((n, n), dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric_d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    maximum_distance = max(float(np.max(positive_values)), global_scale)
    replacement = max(10.0 * maximum_distance, global_scale)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # Robust local scale: the third nearest valid-distance proxy when possible.
    neighborhood_rank = min(2, n - 2)
    local_scale = np.partition(work, neighborhood_rank, axis=1)[:, neighborhood_rank]
    local_scale = np.maximum(local_scale, eps)

    # A second scale captures local density without replacing the robust scale.
    nearest_scale = np.min(work, axis=1)
    nearest_scale = np.maximum(nearest_scale, eps)
    density_scale = np.sqrt(local_scale * nearest_scale)

    edge_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = work / edge_scale
    global_length = work / global_scale

    # Endpoint potential favors edges that are short for both endpoints.
    endpoint_potential = np.sqrt(
        (local_scale[:, None] / np.maximum(work, eps))
        * (local_scale[None, :] / np.maximum(work, eps))
    )

    # Stable endpoint ranks and a smooth mutual-preference score.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = float(n)

    rank_denominator = float(max(n - 1, 1))
    normalized_rank = 0.5 * (
        ranks / rank_denominator + ranks.T / rank_denominator
    )
    rank_affinity = np.exp(-1.15 * normalized_rank)

    # Mutual-neighborhood support gives extra weight to edges selected by both ends.
    candidate_width = min(max(4, n // 5), max(n - 1, 1))
    mutual_support = (
        (ranks < candidate_width) & (ranks.T < candidate_width)
    ).astype(float)
    one_sided_support = (
        (ranks < candidate_width) | (ranks.T < candidate_width)
    ).astype(float)
    neighborhood_support = 1.0 + 0.55 * mutual_support + 0.12 * one_sided_support

    # Avoid strongly mismatched density regions while preserving useful bridge edges.
    scale_mismatch = np.abs(
        np.log(np.maximum(density_scale[:, None], eps))
        - np.log(np.maximum(density_scale[None, :], eps))
    )
    scale_compatibility = np.exp(-0.28 * np.minimum(scale_mismatch, 50.0))

    # Penalize only edges substantially longer than both endpoint neighborhoods.
    endpoint_longness = np.maximum(
        work / np.maximum(local_scale[:, None], eps),
        work / np.maximum(local_scale[None, :], eps),
    )
    excess_longness = np.maximum(endpoint_longness - 1.15, 0.0)

    # Soft hub correction: extremely popular nodes should not dominate sampling.
    local_competition = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        / np.maximum(work, eps)
    ) * np.sqrt(
        np.maximum(local_scale[None, :], eps)
        / np.maximum(work, eps)
    )
    hub_correction = np.minimum(local_competition, 3.0)

    log_score = (
        1.15 * np.log(np.maximum(endpoint_potential, eps))
        + 1.55 * np.log(np.maximum(rank_affinity, eps))
        + 0.55 * np.log(np.maximum(neighborhood_support, eps))
        + 0.35 * np.log(np.maximum(scale_compatibility, eps))
        + 0.18 * np.log(np.maximum(hub_correction, eps))
        - 0.42 * relative_length
        - 0.08 * global_length
        - 0.20 * excess_longness
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

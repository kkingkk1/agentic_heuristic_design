import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-prior matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (d + d.T)
    valid = np.isfinite(symmetric) & (symmetric >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = symmetric[valid & (symmetric > 0.0)]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0
    scale = max(scale, eps)

    safe_distance = np.where(valid, np.maximum(symmetric, 0.0), np.inf)

    finite_valid = safe_distance[valid]
    largest = float(np.max(finite_valid)) if finite_valid.size else scale
    invalid_distance = max(10.0 * largest, scale)

    cost = np.where(valid, safe_distance / scale, invalid_distance / scale)
    cost = np.maximum(cost, 0.0)

    temperature = 0.85
    affinity = np.exp(np.clip(-cost / temperature, -60.0, 0.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_mass = np.sum(affinity, axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, eps)
    transition = affinity / row_mass

    reciprocal_preference = transition * transition.T

    neighborhood_overlap = transition @ transition.T
    neighborhood_overlap = np.maximum(neighborhood_overlap, eps)

    row_norm = np.sqrt(
        np.sum(transition * transition, axis=1, keepdims=True)
    )
    row_norm = np.maximum(row_norm, eps)

    cosine_similarity = (
        transition @ transition.T
    ) / (row_norm @ row_norm.T)
    cosine_similarity = np.clip(cosine_similarity, 0.0, 1.0)

    neighborhood_complement = np.sqrt(
        np.maximum(1.0 - cosine_similarity, eps)
    )

    alternative_cost = cost @ transition.T
    alternative_cost = np.maximum(alternative_cost, 0.0)

    direct_cost = cost
    detour_advantage = np.maximum(
        direct_cost - alternative_cost,
        0.0,
    )

    local_entropy = -np.sum(
        transition * np.log(np.maximum(transition, eps)),
        axis=1,
    )
    entropy_scale = np.maximum(
        np.log(np.maximum(np.sum(valid, axis=1), 2.0)),
        1.0,
    )
    normalized_entropy = local_entropy / entropy_scale

    entropy_compatibility = np.sqrt(
        np.maximum(
            (1.0 - normalized_entropy[:, None])
            * (1.0 - normalized_entropy[None, :]),
            eps,
        )
    )

    log_score = (
        1.35 * np.log(np.maximum(reciprocal_preference, eps))
        + 0.55 * np.log(neighborhood_complement)
        + 0.40 * np.log(entropy_compatibility)
        - 0.85 * np.log1p(3.0 * neighborhood_overlap)
        - 0.30 * detour_advantage
        - 0.18 * direct_cost
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    log_score -= np.max(log_score[finite_scores])
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores], -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

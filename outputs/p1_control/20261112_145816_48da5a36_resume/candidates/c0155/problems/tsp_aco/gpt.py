import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_positive = np.isfinite(d) & (d > 0)
    finite_positive &= ~np.eye(n, dtype=bool)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    valid = finite_positive & finite_positive.T

    positive_values = symmetric_d[valid]
    global_scale = max(float(np.median(positive_values)), eps)

    replacement = max(float(np.max(positive_values)) * 10.0, global_scale)
    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    temperature = max(0.75 * global_scale, eps)

    direct_affinity = np.exp(
        -np.minimum(work / temperature, 745.0)
    )
    direct_affinity[~valid] = 0.0
    np.fill_diagonal(direct_affinity, 0.0)

    row_mass = np.sum(direct_affinity, axis=1, keepdims=True)
    transition = direct_affinity / np.maximum(row_mass, eps)

    two_hop = transition @ transition.T
    two_hop = np.maximum(two_hop, 0.0)

    two_hop_max = float(np.max(two_hop))
    if two_hop_max > 0.0:
        normalized_two_hop = two_hop / two_hop_max
    else:
        normalized_two_hop = np.zeros_like(two_hop)

    endpoint_support = np.sqrt(
        np.maximum(row_mass, eps) * np.maximum(row_mass.T, eps)
    )
    mutual_support = two_hop / np.maximum(endpoint_support, eps)
    support_scale = float(np.max(mutual_support))
    if support_scale > 0.0:
        mutual_support /= support_scale

    exclusivity = normalized_two_hop / (
        1.0 + 0.5 * np.maximum(mutual_support, 0.0)
    )

    log_score = (
        0.55 * np.log(np.maximum(direct_affinity, eps))
        + 0.95 * np.log(np.maximum(normalized_two_hop + eps, eps))
        + 0.80 * np.log(np.maximum(exclusivity + eps, eps))
    )

    valid_scores = log_score[valid]
    log_score -= float(np.max(valid_scores))

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

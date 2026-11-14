import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a global two-hop and diffusion based TSP edge heuristic."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(raw) & (raw > 0.0)
    np.fill_diagonal(finite_positive, False)
    positive = raw[finite_positive]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    d = np.where(np.isfinite(raw), raw, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive_offdiag = work[np.isfinite(work) & (work > 0.0)]
    edge_scale = (
        float(np.median(positive_offdiag))
        if positive_offdiag.size
        else scale
    )
    edge_scale = max(edge_scale, eps)

    affinity = np.exp(
        -np.square(np.minimum(work / edge_scale, 30.0))
    )
    affinity[~np.isfinite(affinity)] = 0.0
    np.fill_diagonal(affinity, 0.0)

    strength = np.sum(affinity, axis=1)
    strength_scale = np.sqrt(
        np.maximum(strength[:, None] * strength[None, :], eps)
    )

    direct_similarity = np.exp(
        -np.square(np.minimum(d / edge_scale, 30.0))
    )
    direct_similarity = np.clip(direct_similarity, 0.0, 1.0)

    common_support = affinity @ affinity.T
    normalized_common = common_support / (strength_scale + eps)
    redundancy_factor = 1.0 / (
        1.0 + 1.8 * np.clip(normalized_common, 0.0, 1.0)
    )

    two_hop = np.min(
        work[:, :, None] + work[None, :, :],
        axis=1,
    )
    two_hop = np.nan_to_num(
        two_hop,
        nan=0.0,
        posinf=edge_scale * 1e6,
        neginf=0.0,
    )

    detour_advantage = (two_hop - d) / (
        two_hop + d + edge_scale * 1e-12
    )
    detour_advantage = np.clip(detour_advantage, 0.0, 1.0)

    endpoint_activity = np.sqrt(
        np.maximum(strength[:, None], eps)
        * np.maximum(strength[None, :], eps)
    )
    activity_factor = endpoint_activity / (
        endpoint_activity + np.median(strength) + eps
    )
    activity_factor = 0.75 + 0.25 * np.clip(activity_factor, 0.0, 1.0)

    score = (
        direct_similarity ** 0.65
        * (0.30 + 0.70 * detour_advantage)
        * redundancy_factor
        * activity_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

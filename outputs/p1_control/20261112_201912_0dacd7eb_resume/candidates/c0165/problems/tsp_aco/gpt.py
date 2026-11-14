import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    d[diagonal] = np.inf

    finite_offdiag = d[~diagonal & np.isfinite(d)]
    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = finite_offdiag[finite_offdiag > 0.0]
    fallback = float(np.max(positive)) if positive.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    d[diagonal] = np.inf

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    neighborhood = np.partition(d, k - 1, axis=1)
    nearest = neighborhood[:, :k]

    weights = np.linspace(1.0, 0.55, k, dtype=float)
    weights /= weights.sum()

    local_scale = np.sum(nearest * weights[None, :], axis=1)
    local_scale = np.maximum(local_scale, eps)

    alternative = np.maximum(neighborhood[:, k - 1], eps)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    ratio_i = d / np.maximum(local_scale[:, None], eps)
    ratio_j = d / np.maximum(local_scale[None, :], eps)
    pair_ratio = d / np.maximum(pair_scale, eps)

    geometric_affinity = np.exp(
        -np.clip(pair_ratio ** 0.85, 0.0, 45.0)
    )

    endpoint_compatibility = np.exp(
        -np.clip(
            0.65 * (np.log1p(ratio_i) + np.log1p(ratio_j))
            + 0.35 * np.abs(np.log1p(ratio_i) - np.log1p(ratio_j)),
            0.0,
            45.0,
        )
    )

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    normalized_rank = ranks / rank_scale

    rank_affinity = np.sqrt(
        1.0 / (1.0 + 2.0 * normalized_rank)
        * 1.0 / (1.0 + 2.0 * normalized_rank.T)
    )

    candidate_width = max(float(k), 1.0)
    candidate_support = np.sqrt(
        np.exp(-np.clip(ranks / candidate_width, 0.0, 45.0))
        * np.exp(-np.clip(ranks.T / candidate_width, 0.0, 45.0))
    )

    mutual_bonus = (
        1.0
        + 0.20
        * (
            (normalized_rank <= 0.25)
            & (normalized_rank.T <= 0.25)
        ).astype(float)
        + 0.10
        * (
            (normalized_rank <= 0.50)
            & (normalized_rank.T <= 0.50)
        ).astype(float)
    )

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = substitute_scale / (
        d + substitute_scale + eps
    )
    margin_bonus = 0.92 + 0.08 * margin

    score = (
        geometric_affinity ** 1.15
        * endpoint_compatibility ** 1.10
        * rank_affinity ** 1.05
        * (0.60 + 0.40 * candidate_support)
        * mutual_bonus
        * margin_bonus
    )

    score = 0.5 * (score + score.T)
    score[diagonal] = 0.0

    off_diagonal = ~diagonal
    values = score[off_diagonal]
    if values.size:
        cutoff = float(np.percentile(values, 85.0))
        score *= np.where(score >= cutoff, 1.025, 1.0)

    score = 0.5 * (score + score.T)
    score[diagonal] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)

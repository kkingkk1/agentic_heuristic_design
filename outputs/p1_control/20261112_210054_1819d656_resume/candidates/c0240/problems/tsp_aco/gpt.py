import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    diagonal = np.eye(n, dtype=bool)
    finite_offdiag = d[~diagonal & np.isfinite(d)]

    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(finite_offdiag, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    positive = finite_offdiag[finite_offdiag > 0.0]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.ceil(1.5 * np.sqrt(n)))))

    nearest = np.partition(d, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = d / (pair_scale + eps)
    relative_affinity = np.exp(
        -np.clip(relative_length ** 0.78, 0.0, 50.0)
    )

    ratio_i = d / local_scale[:, None]
    ratio_j = d / local_scale[None, :]

    endpoint_compatibility = np.exp(
        -0.68
        * (
            np.log1p(np.maximum(ratio_i, 0.0))
            + np.log1p(np.maximum(ratio_j, 0.0))
        )
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_rank = ranks / max(float(n - 1), 1.0)

    rank_affinity = np.exp(
        -1.45
        * (
            normalized_rank
            + normalized_rank.T
        )
    )

    candidate_width = max(float(k) * 1.15, 1.0)
    candidate_support = np.sqrt(
        np.exp(
            -np.clip(
                normalized_rank * (n - 1) / candidate_width,
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                normalized_rank.T * (n - 1) / candidate_width,
                0.0,
                50.0,
            )
        )
    )
    soft_candidate_gate = 0.50 + 0.50 * candidate_support

    alternative_i = np.partition(d, k - 1, axis=1)[:, k - 1]
    alternative_i = np.maximum(alternative_i, eps)

    margin_i = np.clip(
        (alternative_i[:, None] - d)
        / (alternative_i[:, None] + 0.85 * local_scale[:, None] + eps),
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (alternative_i[None, :] - d)
        / (alternative_i[None, :] + 0.85 * local_scale[None, :] + eps),
        0.0,
        1.0,
    )

    margin_bonus = 1.0 + 0.10 * np.sqrt(margin_i * margin_j)

    global_affinity = np.exp(
        -np.clip((d / global_scale) ** 0.68, 0.0, 50.0)
    )
    global_guard = 0.40 + 0.60 * global_affinity

    score = (
        relative_affinity
        * endpoint_compatibility
        * rank_affinity
        * soft_candidate_gate
        * margin_bonus
        * global_guard
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

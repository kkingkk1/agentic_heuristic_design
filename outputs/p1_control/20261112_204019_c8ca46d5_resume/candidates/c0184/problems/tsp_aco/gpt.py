import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

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

    fallback = float(np.max(finite_offdiag))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    positive = finite_offdiag[finite_offdiag > 0.0]
    global_scale = float(np.median(positive)) if positive.size else fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n))) + 2))
    ordered = np.sort(d, axis=1)
    local_neighbors = ordered[:, :k]

    local_scale = np.median(local_neighbors, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    density = global_scale / local_scale
    density = density / max(float(np.median(density)), np.finfo(float).eps)

    density_balance = np.sqrt(
        np.minimum(density[:, None], density[None, :])
        / np.maximum(density[:, None], density[None, :])
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    rank_a = ranks / rank_scale
    rank_b = ranks.T / rank_scale

    mutual_rank = np.exp(-1.85 * (rank_a + rank_b))
    rank_gap = np.abs(rank_a - rank_b)
    rank_agreement = np.exp(-1.35 * rank_gap)

    relative = d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_term = np.exp(-np.clip(relative ** 0.75, 0.0, 50.0))

    ratio_a = d / local_scale[:, None]
    ratio_b = d / local_scale[None, :]
    compatibility = np.exp(
        -0.85
        * (
            np.abs(np.log1p(ratio_a))
            + np.abs(np.log1p(ratio_b))
        )
    )

    global_term = np.exp(-np.clip((d / global_scale) ** 0.9, 0.0, 50.0))

    score = (
        relative_term
        * compatibility
        * (0.40 + 0.60 * global_term)
        * (0.20 + 0.80 * mutual_rank)
        * (0.40 + 0.60 * rank_agreement)
        * (0.50 + 0.50 * density_balance)
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

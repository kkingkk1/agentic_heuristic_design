import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    fallback = float(np.max(positive)) if positive.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    substitute = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / (pair_scale + eps)

    closeness = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    competitiveness = 1.0 / (1.0 + relative_distance)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    substitute_scale = np.sqrt(
        substitute[:, None] * substitute[None, :]
    )
    margin = substitute_scale / (d + substitute_scale + eps)
    margin_factor = 0.92 + 0.08 * margin

    base = (
        closeness
        * (0.60 + 0.40 * mutual_competitiveness)
        * (0.70 + 0.30 * mutual_rank)
        * margin_factor
    )

    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_values = base[upper_i, upper_j]

    if upper_values.size:
        cutoff_index = min(
            upper_values.size - 1,
            max(0, int(np.ceil(0.20 * upper_values.size)) - 1),
        )
        cutoff = float(
            np.partition(upper_values, cutoff_index)[cutoff_index]
        )
        score = base * np.where(base >= cutoff, 1.06, 1.0)
    else:
        score = base

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)

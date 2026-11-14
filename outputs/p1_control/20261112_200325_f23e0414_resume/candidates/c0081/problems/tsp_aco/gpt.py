import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    alternative = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative = d / (pair_scale + eps)

    closeness = np.exp(-relative)
    competitiveness = 1.0 / (1.0 + relative)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    normalized_margin = substitute_scale / (d + eps)
    margin_factor = 1.0 - np.exp(-normalized_margin)

    base = (
        closeness
        * (0.65 + 0.35 * mutual_competitiveness)
        * (0.75 + 0.25 * margin_factor)
    )
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)

    local_order = np.argsort(
        -normalized_margin,
        axis=1,
        kind="mergesort",
    )
    local_rank = np.empty_like(local_order, dtype=float)
    row_index = np.arange(n)[:, None]
    local_rank[row_index, local_order] = np.arange(n, dtype=float)[None, :]

    candidate_count = max(1, int(np.ceil(0.20 * (n - 1))))
    local_high = (
        (local_rank < candidate_count)
        & off_diagonal
    )
    mutual_high_margin = local_high & local_high.T

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_values = base[upper_i, upper_j]

    if upper_values.size:
        cutoff_index = min(
            upper_values.size - 1,
            max(0, int(np.ceil(0.80 * upper_values.size)) - 1),
        )
        cutoff = float(
            np.partition(upper_values, cutoff_index)[cutoff_index]
        )
        top_base_edges = base >= cutoff
    else:
        top_base_edges = np.zeros((n, n), dtype=bool)

    concentrated_edges = mutual_high_margin & top_base_edges

    score = base * np.where(concentrated_edges, 1.09, 1.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(values))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

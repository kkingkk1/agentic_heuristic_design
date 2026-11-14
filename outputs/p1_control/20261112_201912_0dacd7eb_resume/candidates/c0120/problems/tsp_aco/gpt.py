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

    k = max(1, min(n - 1, int(np.ceil(1.25 * np.sqrt(n)))))
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    eps = np.finfo(float).eps
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    alternative = np.maximum(
        np.partition(work, k - 1, axis=1)[:, k - 1],
        eps,
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = d / (pair_scale + eps)

    closeness = np.exp(-1.10 * relative)
    competitiveness = 1.0 / (1.0 + 0.90 * relative)
    mutual_competitiveness = np.sqrt(
        competitiveness * competitiveness.T
    )

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    local_margin = substitute_scale / (d + substitute_scale + eps)
    margin_factor = 0.78 + 0.22 * local_margin

    base = (
        closeness
        * (0.68 + 0.32 * mutual_competitiveness)
        * margin_factor
    )
    base = 0.5 * (base + base.T)
    np.fill_diagonal(base, 0.0)

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_values = base[upper_i, upper_j]

    if upper_values.size:
        cutoff_index = min(
            upper_values.size - 1,
            max(0, int(np.ceil(0.8 * upper_values.size)) - 1),
        )
        cutoff = float(np.partition(upper_values, cutoff_index)[cutoff_index])
        top_edges = base >= cutoff
        score = base * np.where(top_edges, 1.06, 1.0)
    else:
        score = base

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

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

    neighborhood_size = max(
        1,
        min(n - 1, int(np.ceil(1.5 * np.sqrt(n)))),
    )

    neighborhood = np.partition(
        work,
        neighborhood_size - 1,
        axis=1,
    )

    nearest = neighborhood[:, :neighborhood_size]
    local_scale = np.maximum(nearest.mean(axis=1), eps)
    alternative = np.maximum(
        neighborhood[:, neighborhood_size - 1],
        eps,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = d / (pair_scale + eps)
    closeness = np.exp(-relative_length)

    directed_attractiveness = np.exp(
        -d / (local_scale[:, None] + eps)
    )
    np.fill_diagonal(directed_attractiveness, 0.0)

    row_mass = np.maximum(
        directed_attractiveness.sum(axis=1),
        eps,
    )

    directed_competition = (
        directed_attractiveness / row_mass[:, None]
    )

    mutual_competition = np.sqrt(
        directed_competition
        * directed_competition.T
    )

    order = np.argsort(
        work,
        axis=1,
        kind="mergesort",
    )

    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    substitute_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )

    margin = substitute_scale / (
        d + substitute_scale + eps
    )

    bounded_margin = 0.93 + 0.07 * margin

    connectivity = row_mass / np.maximum(
        float(row_mass.max()),
        eps,
    )

    connectivity_factor = 0.97 + 0.03 * np.sqrt(
        connectivity[:, None] * connectivity[None, :]
    )

    score = (
        closeness
        * (0.58 + 0.42 * mutual_competition)
        * (0.72 + 0.28 * mutual_rank)
        * bounded_margin
        * connectivity_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        center = float(np.percentile(values, 75.0))
        spread = float(np.std(values))
        spread = max(spread, eps)

        soft_gate = 1.0 / (
            1.0
            + np.exp(
                -np.clip((score - center) / spread, -30.0, 30.0)
            )
        )

        score *= 1.0 + 0.045 * soft_gate

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

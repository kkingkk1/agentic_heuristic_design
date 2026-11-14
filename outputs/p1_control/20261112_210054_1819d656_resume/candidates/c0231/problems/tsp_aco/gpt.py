import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_positive = distances[
        np.isfinite(distances)
        & (distances > 0.0)
        & off_diagonal
    ]

    if finite_positive.size:
        fallback = float(np.percentile(finite_positive, 90.0))
        global_scale = float(np.median(finite_positive))
        global_upper = float(np.percentile(finite_positive, 75.0))
    else:
        fallback = 1.0
        global_scale = 1.0
        global_upper = 1.0

    fallback = max(fallback, np.finfo(float).eps)
    global_scale = max(global_scale, np.finfo(float).eps)
    global_upper = max(global_upper, global_scale)

    d = np.asarray(distances, dtype=float).copy()
    d = np.nan_to_num(
        d,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.ceil(1.5 * np.sqrt(n)))))

    nearest_values = np.partition(work, k - 1, axis=1)[:, :k]
    nearest_values.sort(axis=1)

    weights = np.linspace(1.0, 0.45, k, dtype=float)
    weights /= np.sum(weights)

    local_mean = np.sum(nearest_values * weights[None, :], axis=1)
    local_median = np.median(nearest_values, axis=1)
    nearest = nearest_values[:, 0]
    alternative = nearest_values[:, min(1, k - 1)]

    local_scale = (
        0.55 * local_mean
        + 0.30 * local_median
        + 0.15 * nearest
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = d / (pair_scale + np.finfo(float).eps)
    local_score = np.exp(
        -np.clip(normalized_distance ** 0.88, 0.0, 50.0)
    )

    median_scale = np.sqrt(
        np.maximum(local_median[:, None], np.finfo(float).eps)
        * np.maximum(local_median[None, :], np.finfo(float).eps)
    )
    median_score = np.exp(
        -np.clip((d / (median_scale + np.finfo(float).eps)) ** 0.72, 0.0, 50.0)
    )

    candidate_indices = np.argpartition(
        work,
        kth=k - 1,
        axis=1,
    )[:, :k]

    candidate_values = np.take_along_axis(
        work,
        candidate_indices,
        axis=1,
    )
    candidate_order = np.argsort(
        candidate_values,
        axis=1,
        kind="mergesort",
    )
    candidate_indices = np.take_along_axis(
        candidate_indices,
        candidate_order,
        axis=1,
    )

    rank_score = np.full((n, n), 1.0 / 3.0, dtype=float)
    rank_denominator = max(float(n - 1), 1.0)

    rank_values = 1.0 / (
        1.0
        + 2.0 * (
            np.arange(k, dtype=float)[None, :]
            / rank_denominator
        )
    )
    rank_score[
        np.arange(n)[:, None],
        candidate_indices,
    ] = rank_values

    mutual_rank = np.sqrt(rank_score * rank_score.T)

    top_fraction = min(0.35, max(1.0 / rank_denominator, 0.25))
    local_rank_fraction = np.zeros((n, n), dtype=float)
    local_rank_fraction[
        np.arange(n)[:, None],
        candidate_indices,
    ] = (
        np.arange(k, dtype=float)[None, :]
        / max(float(k - 1), 1.0)
    )

    mutual_candidate = (
        (local_rank_fraction <= top_fraction)
        & (local_rank_fraction.T <= top_fraction)
    )
    mutual_bonus = np.where(mutual_candidate, 1.14, 1.0)

    alternative_i = alternative[:, None]
    alternative_j = alternative[None, :]
    substitute_scale = np.sqrt(
        alternative_i * alternative_j
    )

    regret = (
        substitute_scale + np.finfo(float).eps
    ) / (
        d
        + substitute_scale
        + np.finfo(float).eps
    )
    regret = np.clip(regret, 0.0, 1.0)

    compatibility_i = d / (
        local_scale[:, None]
        + np.finfo(float).eps
    )
    compatibility_j = d / (
        local_scale[None, :]
        + np.finfo(float).eps
    )

    mismatch = np.abs(
        np.log1p(compatibility_i)
        - np.log1p(compatibility_j)
    )
    endpoint_compatibility = np.exp(
        -np.clip(
            0.48 * (
                np.log1p(compatibility_i)
                + np.log1p(compatibility_j)
            )
            + 0.22 * mismatch,
            0.0,
            50.0,
        )
    )

    global_gate = np.exp(
        -np.clip(
            (d / global_scale) ** 0.68,
            0.0,
            50.0,
        )
    )
    global_gate = 0.40 + 0.60 * global_gate

    tail_penalty = np.maximum(
        d / global_upper - 1.0,
        0.0,
    )
    tail_guard = 1.0 / (
        1.0 + 0.16 * tail_penalty ** 1.20
    )

    score = (
        local_score ** 1.20
        * median_score ** 0.30
        * mutual_rank ** 1.10
        * mutual_bonus
        * endpoint_compatibility ** 1.05
        * global_gate
        * tail_guard
        * (0.90 + 0.16 * regret)
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

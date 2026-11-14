import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, scale-invariant edge-priority matrix for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite_positive = np.isfinite(distances) & (distances > 0)
    valid = finite_positive & finite_positive.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    ordered = np.sort(d, axis=1)

    nearest = ordered[:, 0]
    nearest_safe = np.where(
        np.isfinite(nearest),
        nearest,
        1.0,
    )

    k = min(6, max(1, n - 1))
    local_neighbors = ordered[:, :k]
    finite_neighbor_mask = np.isfinite(local_neighbors)

    neighbor_counts = finite_neighbor_mask.sum(axis=1)
    neighbor_sums = np.where(
        finite_neighbor_mask,
        local_neighbors,
        0.0,
    ).sum(axis=1)

    neighbor_mean = np.divide(
        neighbor_sums,
        np.maximum(neighbor_counts, 1),
    )
    neighbor_mean = np.where(
        neighbor_counts > 0,
        neighbor_mean,
        nearest_safe,
    )

    local_scale = np.sqrt(
        np.maximum(nearest_safe, np.finfo(float).eps)
        * np.maximum(neighbor_mean, np.finfo(float).eps)
    )

    if n > 2:
        second = ordered[:, 1]
        second_safe = np.where(
            np.isfinite(second),
            second,
            nearest_safe,
        )
    else:
        second_safe = nearest_safe

    regret = np.divide(
        second_safe - nearest_safe,
        np.maximum(second_safe, np.finfo(float).eps),
    )
    regret = np.clip(
        np.nan_to_num(regret, nan=0.0),
        0.0,
        1.0,
    )

    relative_length = np.divide(
        d,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full_like(d, np.inf),
        where=np.isfinite(d),
    )

    clipped_relative = np.clip(relative_length, 0.0, 50.0)

    distance_attractiveness = np.exp(
        -np.power(clipped_relative, 1.25)
    )

    row_temperature = np.maximum(
        0.85 * local_scale[:, None],
        np.finfo(float).eps,
    )
    choice_logits = np.divide(
        -d,
        row_temperature,
        out=np.full_like(d, -np.inf),
        where=np.isfinite(d),
    )
    choice_logits = np.clip(choice_logits, -60.0, 0.0)

    choice_weight = np.exp(choice_logits)
    choice_weight[~valid] = 0.0

    choice_normalizer = choice_weight.sum(axis=1, keepdims=True)
    choice_probability = np.divide(
        choice_weight,
        choice_normalizer,
        out=np.zeros_like(choice_weight),
        where=choice_normalizer > 0,
    )

    mutual_choice = np.sqrt(
        choice_probability * choice_probability.T
    )

    row_order = np.argsort(d, axis=1)
    ranks = np.full((n, n), np.inf, dtype=float)
    row_ranks = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = row_ranks[None, :]

    rank_scale = max(1.0, 0.8 * np.sqrt(float(n)))
    rank_support = np.exp(
        -0.6 * (ranks + ranks.T) / rank_scale
    )
    rank_support[~valid] = 0.0

    endpoint_regret = 1.0 + 0.45 * (
        regret[:, None] + regret[None, :]
    )

    density_bonus = 0.70 + 0.30 * mutual_choice
    rank_bonus = 0.70 + 0.30 * rank_support

    long_edge_penalty = np.exp(
        -0.18 * np.maximum(relative_length - 1.0, 0.0)
    )

    score = (
        distance_attractiveness
        * density_bonus
        * rank_bonus
        * endpoint_regret
        * long_edge_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

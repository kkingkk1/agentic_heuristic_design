import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)
    valid = finite & finite.T

    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.10 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.80 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.5
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.28 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.20, 1.0)

    long_edge_penalty = np.exp(
        -0.50 * np.maximum(relative_distance - 1.0, 0.0)
    )

    base_scores = (
        distance_term
        * (0.48 + 0.32 * bidirectional_rank + 0.20 * endpoint_quality)
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    row_mass = np.sum(proximity, axis=1)
    row_mass = np.maximum(row_mass, eps)

    row_probability = proximity / row_mass[:, None]
    mutual_support = np.sqrt(
        row_probability * row_probability.T
    )

    concentration = np.sum(
        row_probability * row_probability,
        axis=1,
    )
    concentration_scale = float(np.median(concentration))
    if not np.isfinite(concentration_scale) or concentration_scale <= 0.0:
        concentration_scale = 1.0

    forcedness = np.sqrt(
        concentration / max(concentration_scale, eps)
    )
    forcedness = np.clip(forcedness, 0.65, 1.60)
    forcedness_pair = 0.5 * (
        forcedness[:, None] + forcedness[None, :]
    )

    support_scale = float(np.median(row_mass[row_mass > 0.0]))
    if not np.isfinite(support_scale) or support_scale <= 0.0:
        support_scale = 1.0

    sparsity = np.sqrt(
        support_scale / np.maximum(row_mass, eps)
    )
    sparsity = np.clip(sparsity, 0.75, 1.55)
    sparsity_pair = 0.5 * (
        sparsity[:, None] + sparsity[None, :]
    )

    profile_norm = np.sqrt(
        np.sum(proximity * proximity, axis=1)
    )
    denominator = profile_norm[:, None] * profile_norm[None, :]
    overlap = np.divide(
        proximity @ proximity.T,
        denominator,
        out=np.zeros((n, n), dtype=float),
        where=denominator > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_signal = np.clip(
        (1.0 - overlap)
        * (1.0 - np.exp(-np.maximum(relative_distance, 0.0))),
        0.0,
        1.0,
    )

    # Preserve edges that are relatively competitive at both endpoints.
    competition = np.sqrt(
        np.maximum(mutual_support, 0.0)
    )
    competition = np.clip(competition, 0.0, 1.0)

    diversity_factor = np.clip(
        1.0
        + 0.045 * (sparsity_pair - 1.0)
        + 0.040 * bridge_signal
        - 0.030 * overlap,
        0.90,
        1.10,
    )

    potential_factor = np.clip(
        1.0
        + 0.050 * (forcedness_pair - 1.0)
        + 0.040 * competition
        + 0.035 * bridge_signal
        - 0.035 * overlap,
        0.90,
        1.12,
    )

    scores = (
        base_scores
        * diversity_factor
        * potential_factor
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    valid_input = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    # Use the best available direction while enforcing an undirected TSP prior.
    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = safe[valid & (safe > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = float(np.median(finite_values))

    global_scale = max(global_scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)

    usable_k = min(4, n - 1)
    nearest_indices = row_order[:, :usable_k]
    nearest_values = np.take_along_axis(
        safe,
        nearest_indices,
        axis=1,
    )

    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_local_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    # Main signal: global and locally calibrated distance quality.
    global_score = np.exp(
        -np.clip(safe / global_scale, 0.0, 40.0)
    )
    local_score = np.exp(
        -np.clip(safe / pair_local_scale, 0.0, 40.0)
    )
    direct_score = 0.35 * global_score + 0.65 * local_score

    # Stable local rank, with a bounded influence.
    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )
    rank_score = np.clip(rank_score, 0.0, 1.0)

    top_k = min(2, n - 1)
    top_two = np.zeros((n, n), dtype=bool)
    top_two[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    mutual_top_two = (top_two & top_two.T & valid).astype(float)

    nearest = row_order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Local completion potential:
    # estimate the quality of the best alternative edge at each endpoint,
    # excluding the candidate edge itself.
    alternative = np.full((n, n), np.inf, dtype=float)

    if n > 2:
        first = row_order[:, 0]
        second = row_order[:, 1]

        best_value = safe[np.arange(n), first]
        second_value = safe[np.arange(n), second]

        for j in range(n):
            use_second = first == j
            alternative[:, j] = np.where(
                use_second,
                second_value,
                best_value,
            )

    alternative = np.where(
        np.isfinite(alternative),
        alternative,
        local_scale[:, None],
    )

    endpoint_completion = 0.5 * (
        np.exp(
            -np.clip(
                alternative / local_scale[:, None],
                0.0,
                20.0,
            )
        )
        + np.exp(
            -np.clip(
                alternative.T / local_scale[None, :],
                0.0,
                20.0,
            )
        )
    )

    # Mild isolation correction, capped to avoid rewarding entropy alone.
    isolation = pair_local_scale / (
        safe + global_scale + eps
    )
    isolation = np.clip(isolation, 0.0, 1.0)
    isolation_bonus = 0.5 + 0.5 * isolation

    scores = (
        0.54 * direct_score
        + 0.18 * rank_score
        + 0.10 * endpoint_completion
        + 0.08 * mutual_nearest
        + 0.05 * mutual_top_two
        + 0.05 * isolation_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)

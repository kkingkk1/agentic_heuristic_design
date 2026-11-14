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

    # Use only mutually available edges when the input is asymmetric.
    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, np.finfo(float).eps)

    # Robust local scale from the closest usable neighbors of each endpoint.
    usable_k = min(5, n - 1)
    sorted_safe = np.sort(safe, axis=1)
    local_values = sorted_safe[:, :usable_k]
    local_finite = np.isfinite(local_values)

    local_sum = np.sum(
        np.where(local_finite, local_values, 0.0),
        axis=1,
    )
    local_count = np.maximum(np.sum(local_finite, axis=1), 1)
    local_scale = local_sum / local_count
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    safe_for_math = np.where(valid, safe, 0.0)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    # Global and endpoint-local distance evidence remain the dominant signals.
    global_distance = np.exp(
        -np.clip(safe_for_math / global_scale, 0.0, 50.0)
    )
    local_distance = np.exp(
        -np.clip(safe_for_math / pair_scale, 0.0, 50.0)
    )
    distance_score = np.sqrt(global_distance * local_distance)

    # Stable reciprocal ranks preserve raw distance ordering within neighborhoods.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[
        np.arange(n)[:, None],
        row_order
    ] = np.arange(n, dtype=float)

    col_rank[
        col_order,
        np.arange(n)[None, :]
    ] = np.arange(n, dtype=float)[:, None]

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    reciprocal_rank = np.clip(reciprocal_rank, 0.0, 1.0)

    # Mutual top-neighborhood edges receive only a bounded reinforcement.
    top_k = min(3, n - 1)
    top_mask = np.zeros((n, n), dtype=bool)
    top_mask[
        np.arange(n)[:, None],
        row_order[:, :top_k]
    ] = True

    mutual_top = (top_mask & top_mask.T & valid).astype(float)

    nearest = np.argmin(safe, axis=1)
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Mild scarcity bonus: retain alternatives around locally sparse endpoints.
    scarcity_i = np.clip(
        local_scale[:, None]
        / (safe_for_math + local_scale[:, None]),
        0.0,
        1.0,
    )
    scarcity_j = np.clip(
        local_scale[None, :]
        / (safe_for_math + local_scale[None, :]),
        0.0,
        1.0,
    )
    scarcity = np.sqrt(scarcity_i * scarcity_j)

    raw_scores = (
        distance_score
        * (0.48 + 0.52 * reciprocal_rank)
        * (1.0 + 0.16 * mutual_top + 0.24 * mutual_nearest)
        * (1.0 + 0.10 * scarcity)
    )

    raw_scores[~valid] = 0.0
    raw_scores = 0.5 * (raw_scores + raw_scores.T)

    # Penalize neighborhood concentration without suppressing any edge outright.
    row_totals = np.sum(raw_scores, axis=1)
    probabilities = np.divide(
        raw_scores,
        row_totals[:, None],
        out=np.zeros_like(raw_scores),
        where=row_totals[:, None] > 0.0,
    )
    concentration = np.sum(probabilities * probabilities, axis=1)
    concentration = np.clip(concentration, 0.0, 1.0)

    diversity_penalty = 1.0 / np.sqrt(
        (1.0 + 0.16 * concentration[:, None])
        * (1.0 + 0.16 * concentration[None, :])
    )

    scores = raw_scores * diversity_penalty
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

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge priorities based on two-step path savings."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    finite_edges = d[valid]
    positive_edges = finite_edges[finite_edges > 0.0]
    eps = np.finfo(float).eps

    if positive_edges.size:
        global_scale = float(np.median(positive_edges))
    else:
        global_scale = float(np.max(finite_edges))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, d, np.inf)

    # Best two-step directed replacement i -> k -> j.
    if n >= 3:
        two_step = np.full((n, n), np.inf, dtype=float)

        for k in range(n):
            left = work[:, k][:, None]
            right = work[k, :][None, :]
            candidate = left + right
            candidate[:, k] = np.inf
            candidate[k, :] = np.inf
            two_step = np.minimum(two_step, candidate)

        two_step[~valid] = np.inf
        direct = np.where(valid, d, np.inf)

        # Positive savings indicate that the direct edge is substantially
        # shorter than every available two-step alternative.
        savings = np.maximum(two_step - direct, 0.0)
        relative_savings = savings / (global_scale + direct + eps)

        # Edges without a finite replacement retain a moderate structural bonus.
        no_replacement = ~np.isfinite(two_step) & valid
        relative_savings[no_replacement] = 1.0
    else:
        relative_savings = np.zeros_like(d, dtype=float)

    normalized_distance = np.divide(
        d,
        global_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )

    # A soft global distance preference prevents structurally special but
    # excessively long edges from dominating.
    distance_preference = np.exp(
        -np.clip(normalized_distance, 0.0, 60.0)
    )

    shortcut_preference = 1.0 - np.exp(
        -np.clip(2.4 * relative_savings, 0.0, 60.0)
    )

    # Keep every valid edge sampleable while emphasizing structural shortcuts.
    score = (
        distance_preference
        * (0.08 + 0.92 * shortcut_preference)
    )

    # If no two-step information exists, fall back to a globally scaled
    # inverse-distance preference.
    if n < 3:
        score = 0.12 + distance_preference

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    totals = score.sum(axis=1, keepdims=True)
    normalized = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    # Rows with valid edges but numerical underflow receive uniform weights.
    empty_rows = (totals[:, 0] <= 0.0)
    for i in np.flatnonzero(empty_rows):
        candidates = valid[i]
        count = int(np.count_nonzero(candidates))
        if count:
            normalized[i, candidates] = 1.0 / count

    return normalized

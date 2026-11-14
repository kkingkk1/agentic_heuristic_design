import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return triangle-savings-based stochastic edge-priority indicators."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros_like(distances, dtype=float)

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    pairwise = np.where(
        valid,
        0.5 * (distances + distances.T),
        np.inf,
    )

    positive = pairwise[valid & (pairwise > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_values = np.where(valid, pairwise, np.nan)
    row_median = np.nanmedian(row_values, axis=1)
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0.0),
        row_median,
        scale,
    )

    endpoint_scale = np.sqrt(
        np.maximum(row_median[:, None] * row_median[None, :], scale**2)
    )

    relative_cost = pairwise / endpoint_scale
    relative_cost = np.maximum(relative_cost, 0.0)

    # Algebraic distance decay, avoiding a sharply concentrated Gibbs prior.
    distance_term = 1.0 / (1.0 + relative_cost**1.35)

    # Compute the cheapest two-edge detour through a third node.
    detour = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left = pairwise[:, k][:, None]
        right = pairwise[k, :][None, :]
        candidate = left + right

        candidate[k, :] = np.inf
        candidate[:, k] = np.inf
        detour = np.minimum(detour, candidate)

    usable_detour = valid & np.isfinite(detour)
    savings = np.zeros((n, n), dtype=float)
    savings[usable_detour] = (
        detour[usable_detour] - pairwise[usable_detour]
    )

    normalized_savings = np.clip(savings / scale, -4.0, 4.0)

    # Positive savings identify edges that cannot be cheaply replaced
    # by a two-step connection through another node.
    bridge_term = 0.5 + 0.5 * (
        normalized_savings / (1.0 + np.abs(normalized_savings))
    )
    bridge_term[~valid] = 0.0

    # Retain a weak endpoint-distinctiveness signal.
    endpoint_term = 1.0 / (
        1.0
        + np.sqrt(
            np.maximum(
                pairwise / np.maximum(row_median[:, None], scale * 1e-12),
                0.0,
            )
            * np.maximum(
                pairwise / np.maximum(row_median[None, :], scale * 1e-12),
                0.0,
            )
        )
    )

    score = distance_term * (0.65 + 0.70 * bridge_term) * (
        0.75 + 0.25 * endpoint_term
    )
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    mean_score = float(np.mean(score[valid])) if np.any(valid) else 0.0
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)

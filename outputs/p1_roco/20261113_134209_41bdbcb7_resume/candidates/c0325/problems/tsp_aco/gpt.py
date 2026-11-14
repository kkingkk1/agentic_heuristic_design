import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return result

    forward = np.where(directed_valid, distances, np.inf)
    backward = forward.T

    safe = np.minimum(forward, backward)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return result

    safe = np.where(valid, safe, np.inf)
    np.fill_diagonal(safe, np.inf)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    positive_reference = (
        float(np.percentile(positive_values, 75))
        if positive_values.size
        else global_scale
    )
    positive_reference = max(positive_reference, global_scale * 0.25, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        direct_cost = np.divide(
            safe,
            positive_reference,
            out=np.full((n, n), 50.0, dtype=float),
            where=valid,
        )
        direct_cost = np.clip(direct_cost, 0.0, 50.0)

        inverse_cost = 1.0 / (1.0 + direct_cost)
        base_score = inverse_cost ** 1.35

    two_hop_best = np.full((n, n), np.inf, dtype=float)

    for midpoint in range(n):
        left = safe[:, midpoint][:, None]
        right = safe[midpoint, :][None, :]
        route = left + right
        two_hop_best = np.minimum(two_hop_best, route)

    has_two_hop = np.isfinite(two_hop_best) & valid

    detour_ratio = np.ones((n, n), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        detour_ratio[has_two_hop] = np.divide(
            two_hop_best[has_two_hop],
            np.maximum(safe[has_two_hop], eps),
        )

    detour_ratio = np.clip(detour_ratio, 1.0, 12.0)

    with np.errstate(over="ignore"):
        detour_bonus = np.exp(
            0.30 * np.log1p(detour_ratio - 1.0)
        )

    alternate_count = np.zeros((n, n), dtype=float)
    for midpoint in range(n):
        route = (
            np.isfinite(safe[:, midpoint])[:, None]
            & np.isfinite(safe[midpoint, :])[None, :]
        )
        alternate_count += route.astype(float)

    alternate_support = np.clip(
        alternate_count / max(float(n), 1.0),
        0.0,
        1.0,
    )

    support_factor = 0.90 + 0.20 * alternate_support

    scores = (
        base_score
        * detour_bonus
        * support_factor
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

    if np.any(scores > 0.0):
        target_mass = float(np.mean(np.sum(scores, axis=1)))
        scale = np.ones(n, dtype=float)

        for _ in range(24):
            row_mass = scores.sum(axis=1) * scale
            row_mass = np.maximum(row_mass, eps)

            correction = np.sqrt(
                np.divide(
                    target_mass,
                    row_mass,
                    out=np.ones(n, dtype=float),
                    where=row_mass > 0.0,
                )
            )
            correction = np.clip(correction, 0.25, 4.0)
            scale *= correction
            scale = np.clip(scale, 1e-4, 1e4)

        scores *= scale[:, None] * scale[None, :]

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

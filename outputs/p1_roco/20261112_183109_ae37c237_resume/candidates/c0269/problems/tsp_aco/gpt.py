import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)
    valid = finite & finite.T

    if not np.any(valid):
        return result

    with np.errstate(all="ignore"):
        d = 0.5 * (distances + distances.T)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    elif values.size:
        scale = float(np.max(values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    normalized = np.divide(
        d,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = 1.0 / (
        1.0
        + normalized[valid] ** 1.35
    )

    degree_mass = np.sum(affinity, axis=1)
    safe_mass = np.maximum(degree_mass, eps)

    probability = np.divide(
        affinity,
        safe_mass[:, None],
        out=np.zeros_like(affinity),
        where=safe_mass[:, None] > 0.0,
    )

    endpoint_pressure = np.sqrt(
        probability * probability.T
    )

    positive_probability = probability > 0.0
    log_probability = np.log(
        np.maximum(probability, eps)
    )

    entropy = -np.sum(
        np.where(positive_probability, probability * log_probability, 0.0),
        axis=1,
    )

    degree = np.sum(valid, axis=1).astype(float)
    entropy_scale = np.log(np.maximum(degree, 2.0))
    normalized_entropy = np.divide(
        entropy,
        entropy_scale,
        out=np.zeros_like(entropy),
        where=entropy_scale > 0.0,
    )
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

    concentration = 1.0 - normalized_entropy
    concentration_pair = 0.5 * (
        concentration[:, None] + concentration[None, :]
    )

    if n > 2:
        two_best = np.partition(
            affinity,
            kth=min(1, n - 1),
            axis=1,
        )[:, -2:]
        strongest = two_best[:, 1]
        backup = two_best[:, 0]
    else:
        strongest = affinity.max(axis=1)
        backup = np.zeros(n, dtype=float)

    reserve = np.divide(
        backup,
        np.maximum(strongest, eps),
    )
    reserve = np.clip(reserve, 0.0, 1.0)

    edge_competition = np.divide(
        affinity,
        np.sqrt(safe_mass[:, None] * safe_mass[None, :]),
        out=np.zeros_like(affinity),
        where=(safe_mass[:, None] * safe_mass[None, :]) > 0.0,
    )

    if n > 2:
        normalized_affinity = affinity / np.maximum(
            strongest[:, None],
            eps,
        )

        detour_support = np.zeros((n, n), dtype=float)

        for h in range(n):
            if h >= n:
                break
            via_h = np.minimum(
                normalized_affinity[:, h][:, None],
                normalized_affinity[h, :][None, :],
            )
            detour_support = np.maximum(detour_support, via_h)

        detour_support[~valid] = 0.0
        np.fill_diagonal(detour_support, 0.0)
    else:
        detour_support = np.zeros((n, n), dtype=float)

    direct_strength = np.sqrt(
        np.maximum(
            normalized_affinity * normalized_affinity.T,
            0.0,
        )
    )
    direct_strength = np.clip(direct_strength, 0.0, 1.0)

    closure_penalty = 1.0 - 0.12 * (
        detour_support
        * direct_strength
        * (0.5 + 0.5 * endpoint_pressure)
    )
    closure_penalty = np.clip(closure_penalty, 0.86, 1.0)

    reserve_pair = np.sqrt(
        np.maximum(reserve[:, None] * reserve[None, :], 0.0)
    )

    concentration_bonus = 1.0 + 0.10 * concentration_pair
    reserve_bonus = 1.0 + 0.16 * reserve_pair
    pressure_bonus = 1.0 + 0.22 * edge_competition

    affinity_term = np.sqrt(
        np.maximum(
            probability * probability.T,
            0.0,
        )
    )

    scores = (
        affinity_term
        * (0.72 + 0.28 * endpoint_pressure)
        * concentration_bonus
        * reserve_bonus
        * pressure_bonus
        * closure_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)

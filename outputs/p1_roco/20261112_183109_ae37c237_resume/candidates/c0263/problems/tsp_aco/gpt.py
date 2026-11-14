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
        symmetric = 0.5 * (distances + distances.T)

    values = symmetric[valid]
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

    maximum = float(np.max(values))
    filler = max(maximum, scale) * 8.0 + scale

    work = np.where(valid, symmetric, filler)
    np.fill_diagonal(work, filler)

    safe_scale = scale + eps

    # Bounded direct-edge affinity.
    normalized = work / safe_scale
    direct_affinity = 1.0 / (1.0 + np.maximum(normalized, 0.0))
    direct_affinity[~valid] = 0.0
    np.fill_diagonal(direct_affinity, 0.0)

    # Normalize each endpoint's available connection mass.
    row_mass = np.sum(direct_affinity, axis=1)
    row_mass = np.maximum(row_mass, eps)

    endpoint_share = direct_affinity / row_mass[:, None]
    endpoint_share[~valid] = 0.0

    # Edges that are strong choices for both endpoints receive a mild bonus.
    bilateral_share = np.sqrt(
        np.maximum(endpoint_share * endpoint_share.T, 0.0)
    )

    # Compute the best two-hop detour through an intermediate node.
    if n > 2:
        detour = np.full((n, n), filler * 2.0, dtype=float)

        for hub in range(n):
            candidate = work[:, hub][:, None] + work[hub, :][None, :]
            detour = np.minimum(detour, candidate)

        detour[np.eye(n, dtype=bool)] = filler * 2.0
    else:
        detour = np.full((n, n), filler * 2.0, dtype=float)

    detour[~valid] = filler * 2.0

    # Positive values indicate that the direct edge beats its best detour.
    advantage = detour - work
    advantage = np.maximum(advantage, 0.0)

    advantage_scale = float(np.median(advantage[valid]))
    if not np.isfinite(advantage_scale) or advantage_scale <= eps:
        advantage_scale = scale

    direct_gain = 1.0 - np.exp(
        -np.clip(advantage / max(advantage_scale, eps), 0.0, 12.0)
    )
    direct_gain = np.clip(direct_gain, 0.0, 1.0)

    # Penalize edges that have many nearly equivalent alternatives.
    alternative_mass = np.sum(
        direct_affinity[:, :, None] * direct_affinity[None, :, :],
        axis=2,
    )
    alternative_mass = np.maximum(alternative_mass, 0.0)

    self_product = direct_affinity * direct_affinity
    exclusivity = np.divide(
        self_product,
        self_product + alternative_mass + eps,
        out=np.zeros_like(self_product),
        where=valid,
    )
    exclusivity = np.clip(exclusivity, 0.0, 1.0)

    # Preserve a strong distance component while using distinct structural terms.
    distance_component = np.exp(
        -np.clip(work / (1.35 * scale), 0.0, 50.0)
    )

    scores = (
        distance_component
        * (0.58 + 0.22 * bilateral_share)
        * (1.0 + 0.22 * direct_gain)
        * (1.0 + 0.16 * exclusivity)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum_score = float(np.max(scores))
    if np.isfinite(maximum_score) and maximum_score > 0.0:
        scores /= maximum_score

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)

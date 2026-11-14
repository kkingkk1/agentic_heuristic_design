import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural edge-priority indicators for stochastic TSP sampling."""
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

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    normalized = np.divide(
        d,
        scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_percentile = np.zeros((n, n), dtype=float)
    col_percentile = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_values = normalized[i, valid[i]]
        if row_values.size:
            ordered = np.sort(row_values)
            row_percentile[i, valid[i]] = np.searchsorted(
                ordered,
                normalized[i, valid[i]],
                side="right",
            ) / float(ordered.size)

    for j in range(n):
        col_values = normalized[valid[:, j], j]
        if col_values.size:
            ordered = np.sort(col_values)
            col_percentile[valid[:, j], j] = np.searchsorted(
                ordered,
                normalized[valid[:, j], j],
                side="right",
            ) / float(ordered.size)

    bilateral_percentile = np.sqrt(
        (1.0 - row_percentile) * (1.0 - col_percentile)
    )

    profile = np.where(valid, normalized, 0.0)
    profile_scale = np.maximum(
        np.sum(valid, axis=1, keepdims=True),
        1.0,
    )
    profile = profile / profile_scale

    profile_norm = np.sqrt(np.sum(profile * profile, axis=1))
    profile_denominator = profile_norm[:, None] * profile_norm[None, :]
    profile_similarity = np.divide(
        profile @ profile.T,
        profile_denominator,
        out=np.zeros((n, n), dtype=float),
        where=profile_denominator > 0.0,
    )
    profile_difference = np.clip(1.0 - profile_similarity, 0.0, 1.0)

    two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left = normalized[:, k][:, None]
        right = normalized[k, :][None, :]
        candidate = left + right
        usable = valid[:, k][:, None] & valid[k, :][None, :]
        candidate[~usable] = np.inf
        two_hop = np.minimum(two_hop, candidate)

    detour_ratio = np.divide(
        two_hop,
        np.maximum(normalized, np.finfo(float).eps),
        out=np.ones((n, n), dtype=float),
        where=valid & np.isfinite(two_hop),
    )
    detour_advantage = np.clip(
        1.0 - np.divide(
            1.0,
            np.maximum(detour_ratio, 1.0),
        ),
        0.0,
        1.0,
    )

    endpoint_access = np.sum(
        valid & (normalized <= 1.0),
        axis=1,
        dtype=float,
    )
    access_reference = float(np.median(endpoint_access[endpoint_access > 0]))
    if not np.isfinite(access_reference) or access_reference <= 0.0:
        access_reference = 1.0

    endpoint_rarity = np.clip(
        access_reference / np.maximum(endpoint_access, 1.0),
        0.65,
        1.60,
    )
    endpoint_factor = np.sqrt(
        endpoint_rarity[:, None] * endpoint_rarity[None, :]
    )

    scores = (
        0.45 * bilateral_percentile
        + 0.25 * detour_advantage
        + 0.18 * profile_difference
        + 0.12 * np.clip(endpoint_factor / 1.60, 0.0, 1.0)
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

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed neighborhood-consistency priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    valid_values = raw[valid]
    positive = valid_values[valid_values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = max(
        np.finfo(float).eps * max(scale, 1.0),
        1e-12,
    )

    finite_max = float(np.max(valid_values))
    replacement = finite_max + 20.0 * scale

    distances = np.where(valid, raw, replacement).astype(float)
    np.fill_diagonal(distances, replacement)

    valid_count = valid.sum(axis=1)
    safe_count = np.maximum(valid_count, 1)

    sorted_distances = np.sort(distances, axis=1)
    nearest = sorted_distances[:, 0]

    local_index = np.minimum(
        np.maximum((valid_count - 1) // 2, 0),
        n - 1,
    )
    local_reference = sorted_distances[
        np.arange(n),
        local_index,
    ]

    local_bandwidth = np.maximum(
        local_reference - nearest,
        0.20 * scale,
    )
    local_bandwidth = np.maximum(local_bandwidth, eps)

    relative_distance = np.maximum(
        distances - nearest[:, None],
        0.0,
    )

    local_kernel = np.exp(
        -np.minimum(
            relative_distance / local_bandwidth[:, None],
            60.0,
        )
    )
    local_kernel[~valid] = 0.0

    row_mass = local_kernel.sum(axis=1, keepdims=True)
    row_probability = local_kernel / np.maximum(
        row_mass,
        eps,
    )
    row_probability[~valid] = 0.0

    incoming_kernel = local_kernel.T
    incoming_mass = incoming_kernel.sum(
        axis=1,
        keepdims=True,
    )
    incoming_probability = incoming_kernel / np.maximum(
        incoming_mass,
        eps,
    )
    incoming_probability = incoming_probability.T
    incoming_probability[~valid] = 0.0

    direct_probability = np.sqrt(
        np.maximum(
            row_probability * incoming_probability,
            0.0,
        )
    )

    outgoing_overlap = row_probability @ row_probability.T
    incoming_overlap = (
        incoming_probability.T @ incoming_probability
    )

    outgoing_diagonal = np.sqrt(
        np.maximum(
            np.diag(outgoing_overlap),
            eps,
        )
    )
    incoming_diagonal = np.sqrt(
        np.maximum(
            np.diag(incoming_overlap),
            eps,
        )
    )

    outgoing_coherence = outgoing_overlap / np.maximum(
        outgoing_diagonal[:, None] * outgoing_diagonal[None, :],
        eps,
    )
    incoming_coherence = incoming_overlap / np.maximum(
        incoming_diagonal[:, None] * incoming_diagonal[None, :],
        eps,
    )

    outgoing_coherence = np.clip(
        outgoing_coherence,
        0.0,
        1.0,
    )
    incoming_coherence = np.clip(
        incoming_coherence,
        0.0,
        1.0,
    )

    row_entropy = -np.sum(
        row_probability
        * np.log(np.maximum(row_probability, eps)),
        axis=1,
    )
    col_entropy = -np.sum(
        incoming_probability
        * np.log(np.maximum(incoming_probability, eps)),
        axis=1,
    )

    entropy_scale = np.log(
        np.maximum(
            np.maximum(valid_count, 2),
            2,
        )
    )

    row_constrained = np.exp(
        -row_entropy / np.maximum(entropy_scale, eps)
    )
    col_constrained = np.exp(
        -col_entropy / np.maximum(entropy_scale, eps)
    )

    row_constrained /= max(
        float(np.mean(row_constrained)),
        eps,
    )
    col_constrained /= max(
        float(np.mean(col_constrained)),
        eps,
    )

    row_constrained = np.clip(
        row_constrained,
        0.55,
        1.80,
    )
    col_constrained = np.clip(
        col_constrained,
        0.55,
        1.80,
    )

    endpoint_constraint = np.sqrt(
        np.maximum(
            row_constrained[:, None]
            * col_constrained[None, :],
            0.0,
        )
    )

    local_median = np.median(
        np.where(valid, distances, np.nan),
        axis=1,
    )
    local_median = np.nan_to_num(
        local_median,
        nan=scale,
        posinf=scale,
        neginf=scale,
    )

    distance_margin = (
        local_median[:, None] - distances
    ) / np.maximum(local_bandwidth[:, None], eps)

    distance_quality = 1.0 / (
        1.0 + np.exp(
            -np.clip(distance_margin, -40.0, 40.0)
        )
    )
    distance_quality[~valid] = 0.0

    mutual_neighborhood = np.sqrt(
        np.maximum(
            outgoing_coherence * incoming_coherence,
            0.0,
        )
    )

    structural_support = (
        0.50 * mutual_neighborhood
        + 0.30 * direct_probability
        + 0.20 * np.sqrt(
            np.maximum(
                row_probability
                * row_probability.T,
                0.0,
            )
        )
    )

    score = (
        np.maximum(distance_quality, eps) ** 0.34
        * np.maximum(direct_probability, eps) ** 0.26
        * np.maximum(structural_support, eps) ** 0.28
        * np.maximum(endpoint_constraint, eps) ** 0.12
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        fallback = valid.astype(float)
        np.fill_diagonal(fallback, 0.0)
        return fallback

    score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

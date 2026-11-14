import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-consistency edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (
        distances[valid] + distances.T[valid]
    )
    np.fill_diagonal(symmetric, np.inf)

    edge_values = symmetric[valid]
    positive = edge_values[edge_values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(
            np.median(np.abs(positive - median))
        )
        scale = max(
            mad,
            0.25 * median,
            np.finfo(float).eps,
        )
    else:
        scale = 1.0

    finite_values = edge_values[np.isfinite(edge_values)]
    upper = (
        float(np.quantile(finite_values, 0.95))
        if finite_values.size
        else scale
    )
    replacement = max(
        upper + 8.0 * scale,
        scale,
    )

    clean = np.where(
        valid,
        symmetric,
        replacement,
    ).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    positive_clean = np.where(
        valid,
        clean,
        np.nan,
    )

    local_scales = np.empty(n, dtype=float)

    for i in range(n):
        row = clean[i, valid[i]]
        row = row[np.isfinite(row)]

        if row.size == 0:
            local_scales[i] = safe_scale
            continue

        local_median = float(np.median(row))
        local_deviation = float(
            np.median(np.abs(row - local_median))
        )

        local_scales[i] = max(
            0.5 * local_median,
            local_deviation,
            0.12 * safe_scale,
            eps,
        )

    pair_scale = np.sqrt(
        np.maximum(
            local_scales[:, None]
            * local_scales[None, :],
            eps,
        )
    )

    normalized_distance = np.minimum(
        clean / pair_scale,
        50.0,
    )

    affinity = np.exp(-normalized_distance)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_degree = affinity.sum(axis=1)
    column_degree = affinity.sum(axis=0)

    row_degree = np.maximum(row_degree, eps)
    column_degree = np.maximum(column_degree, eps)

    transition = affinity / row_degree[:, None]
    reverse_transition = affinity / column_degree[None, :]

    diffusion = transition @ reverse_transition
    diffusion = np.maximum(diffusion, 0.0)
    diffusion[~valid] = 0.0
    np.fill_diagonal(diffusion, 0.0)

    diffusion_max = float(np.max(diffusion))
    if np.isfinite(diffusion_max) and diffusion_max > 0.0:
        diffusion /= diffusion_max

    common_mass = affinity @ affinity.T
    common_mass = np.maximum(common_mass, 0.0)
    common_mass[~valid] = 0.0
    np.fill_diagonal(common_mass, 0.0)

    common_norm = np.sqrt(
        np.maximum(
            row_degree[:, None] * row_degree[None, :],
            eps,
        )
    )
    common_support = common_mass / common_norm
    common_support[~valid] = 0.0

    support_max = float(np.max(common_support))
    if np.isfinite(support_max) and support_max > 0.0:
        common_support /= support_max

    endpoint_confidence = np.sqrt(
        np.maximum(
            affinity / np.maximum(
                affinity.max(axis=1, keepdims=True),
                eps,
            ),
            0.0,
        )
        * np.maximum(
            affinity / np.maximum(
                affinity.max(axis=0, keepdims=True),
                eps,
            ),
            0.0,
        )
    )
    endpoint_confidence[~valid] = 0.0

    local_entropy = np.zeros(n, dtype=float)

    for i in range(n):
        probabilities = transition[i]
        probabilities = probabilities[probabilities > 0.0]

        if probabilities.size:
            entropy = -float(
                np.sum(
                    probabilities
                    * np.log(
                        np.maximum(probabilities, eps)
                    )
                )
            )
            maximum_entropy = np.log(
                max(float(probabilities.size), 2.0)
            )
            local_entropy[i] = entropy / max(
                maximum_entropy,
                eps,
            )

    selectivity = 1.0 - np.sqrt(
        np.maximum(
            local_entropy[:, None]
            * local_entropy[None, :],
            0.0,
        )
    )
    selectivity = 0.25 + 0.75 * selectivity
    selectivity[~valid] = 0.0

    direct = np.sqrt(
        np.maximum(affinity * endpoint_confidence, 0.0)
    )

    diffusion_signal = np.sqrt(
        np.maximum(
            diffusion * (0.35 + 0.65 * common_support),
            0.0,
        )
    )

    score = (
        0.46 * direct
        + 0.30 * diffusion_signal
        + 0.16 * common_support
        + 0.08 * selectivity
    )

    score *= 0.72 + 0.28 * endpoint_confidence
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

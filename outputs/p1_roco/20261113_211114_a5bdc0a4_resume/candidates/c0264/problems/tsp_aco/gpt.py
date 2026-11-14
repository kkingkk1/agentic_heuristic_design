import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-spectral edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (
        distances[valid] + distances.T[valid]
    )
    np.fill_diagonal(symmetric, np.inf)

    values = symmetric[valid]
    positive = values[values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        spread = float(q75 - q25)
        scale = spread if spread > 0.0 else float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    temperature = max(0.55 * scale, eps)

    safe_distance = np.where(valid, symmetric, 0.0)
    normalized = np.minimum(safe_distance / temperature, 80.0)

    weight = np.zeros((n, n), dtype=float)
    weight[valid] = np.exp(-normalized[valid])

    row_mass = weight.sum(axis=1)
    degree_scale = np.sqrt(
        np.maximum(row_mass[:, None] * row_mass[None, :], eps)
    )

    normalized_weight = np.zeros_like(weight)
    normalized_weight[valid] = (
        weight[valid] / degree_scale[valid]
    )

    normalized_weight = 0.5 * (
        normalized_weight + normalized_weight.T
    )
    normalized_weight[~valid] = 0.0

    degree = normalized_weight.sum(axis=1)
    transition = normalized_weight / np.maximum(
        degree[:, None],
        eps,
    )

    transition = np.nan_to_num(
        transition,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    transition[~valid] = 0.0

    walk2 = transition @ transition
    walk3 = walk2 @ transition

    loop_support = np.sqrt(
        np.maximum(walk2 * walk2.T, 0.0)
    )
    loop_support += 0.5 * np.sqrt(
        np.maximum(walk3 * walk3.T, 0.0)
    )
    loop_support[~valid] = 0.0

    graph_degree = normalized_weight.sum(axis=1)
    laplacian = (
        np.diag(graph_degree)
        - normalized_weight
    )

    try:
        laplacian_pinv = np.linalg.pinv(
            laplacian,
            hermitian=True,
        )
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance[~valid] = 0.0

    finite_resistance = resistance[valid]
    resistance_scale = (
        float(np.median(finite_resistance))
        if finite_resistance.size
        else 1.0
    )
    resistance_scale = max(resistance_scale, eps)

    bridge_signal = 1.0 - np.exp(
        -resistance / resistance_scale
    )
    bridge_signal[~valid] = 0.0

    local_entropy = np.zeros(n, dtype=float)
    for i in range(n):
        row = transition[i]
        mass = float(np.sum(row))
        if mass > eps:
            probabilities = row[row > 0.0]
            local_entropy[i] = -float(
                np.sum(probabilities * np.log(probabilities))
            )

    entropy_scale = max(
        float(np.median(local_entropy[local_entropy > 0.0]))
        if np.any(local_entropy > 0.0)
        else 1.0,
        eps,
    )

    entropy_factor = 1.0 / np.sqrt(
        1.0 + local_entropy / entropy_scale
    )
    endpoint_factor = np.sqrt(
        np.maximum(
            entropy_factor[:, None] * entropy_factor[None, :],
            eps,
        )
    )

    kernel_signal = np.sqrt(
        np.maximum(weight * normalized_weight, 0.0)
    )
    kernel_signal[~valid] = 0.0

    loop_max = float(np.max(loop_support))
    if np.isfinite(loop_max) and loop_max > 0.0:
        loop_signal = loop_support / loop_max
    else:
        loop_signal = np.zeros_like(loop_support)

    resistance_max = float(np.max(bridge_signal))
    if np.isfinite(resistance_max) and resistance_max > 0.0:
        bridge_signal = bridge_signal / resistance_max

    score = (
        np.maximum(kernel_signal, eps) ** 0.80
        * np.maximum(
            0.18 + 0.82 * loop_signal,
            eps,
        ) ** 1.20
        * np.maximum(
            0.72 + 0.28 * bridge_signal,
            eps,
        ) ** 0.75
        * np.maximum(
            0.70 + 0.30 * endpoint_factor,
            eps,
        )
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score = valid.astype(float)
        np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic multi-scale transition priors for stochastic TSP."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = raw[valid]
    symmetric = np.zeros((n, n), dtype=float)
    symmetric[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    symmetric = np.maximum(symmetric, 0.0)

    positive = observed[observed > 0.0]
    median_value = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )

    deviations = np.abs(observed - median_value)
    mad = float(np.median(deviations)) if deviations.size else 0.0
    robust_scale = max(
        mad * 1.4826,
        0.18 * median_value,
        np.finfo(float).eps,
    )

    maximum = float(np.max(symmetric[valid]))
    if not np.isfinite(maximum):
        maximum = median_value

    penalty = maximum + 20.0 * robust_scale
    distances = np.where(valid, symmetric, penalty)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps

    valid_counts = np.maximum(valid.sum(axis=1), 1)
    row_values = np.where(valid, distances, penalty)

    sorted_values = np.sort(row_values, axis=1)
    best = sorted_values[:, 0]
    second = sorted_values[:, 1] if n > 2 else best
    third = sorted_values[:, 2] if n > 3 else second

    local_band = np.maximum(
        third - best,
        0.24 * robust_scale,
    )
    local_band = np.maximum(local_band, eps)

    centered = (
        distances - best[:, None]
    ) / local_band[:, None]

    short_signal = np.zeros((n, n), dtype=float)
    short_signal[valid] = 1.0 / (
        1.0 + np.exp(
            np.clip(centered[valid], -40.0, 40.0)
        )
    )

    temperatures = np.array(
        [
            0.42 * robust_scale,
            0.85 * robust_scale,
            1.55 * robust_scale,
        ],
        dtype=float,
    )

    transition = np.zeros((n, n), dtype=float)
    for temperature in temperatures:
        logits = np.zeros((n, n), dtype=float)
        logits[valid] = -(
            distances[valid] - best.repeat(n)
            if False
            else distances[valid]
        ) / max(float(temperature), eps)

        logits = np.where(
            valid,
            np.clip(
                -distances / max(float(temperature), eps),
                -60.0,
                0.0,
            ),
            -np.inf,
        )

        row_max = np.max(logits, axis=1, keepdims=True)
        weights = np.exp(
            np.where(
                valid,
                logits - row_max,
                -np.inf,
            )
        )
        weights[~valid] = 0.0

        row_sum = np.maximum(
            weights.sum(axis=1, keepdims=True),
            eps,
        )
        transition += weights / row_sum

    transition /= float(len(temperatures))

    mutual_transition = np.sqrt(
        np.maximum(
            transition * transition.T,
            0.0,
        )
    )
    mutual_transition[~valid] = 0.0

    local_mass = transition.sum(axis=1)
    local_mass = np.maximum(local_mass, eps)

    concentration = np.sum(
        transition * transition,
        axis=1,
    )
    concentration /= np.maximum(
        valid_counts.astype(float),
        1.0,
    )
    concentration = np.sqrt(
        np.maximum(concentration, 0.0)
    )

    endpoint_density = np.sqrt(
        np.maximum(
            local_mass[:, None] * local_mass[None, :],
            0.0,
        )
    )
    endpoint_density /= max(
        float(np.max(endpoint_density)),
        eps,
    )

    endpoint_concentration = np.sqrt(
        np.maximum(
            concentration[:, None] * concentration[None, :],
            0.0,
        )
    )

    opportunity = np.zeros((n, n), dtype=float)
    for axis in (0, 1):
        rank_values = np.argsort(
            distances,
            axis=axis,
            kind="stable",
        )
        rank = np.empty((n, n), dtype=float)

        if axis == 1:
            rank[
                np.arange(n)[:, None],
                rank_values,
            ] = np.arange(n, dtype=float)
        else:
            rank[
                rank_values,
                np.arange(n)[None, :],
            ] = np.arange(n, dtype=float)

        rank_scale = max(
            1.0,
            0.75 * np.sqrt(float(n)),
        )
        rank_weight = np.exp(-rank / rank_scale)
        rank_weight[~valid] = 0.0
        opportunity += rank_weight

    opportunity /= 2.0
    opportunity_max = float(np.max(opportunity))
    if np.isfinite(opportunity_max) and opportunity_max > 0.0:
        opportunity /= opportunity_max

    global_order = np.sort(symmetric[valid])
    global_rank = np.searchsorted(
        global_order,
        symmetric,
        side="right",
    ).astype(float)
    global_rank = 1.0 - (
        global_rank / max(float(global_order.size), 1.0)
    )
    global_rank = np.clip(global_rank, 0.0, 1.0)
    global_rank[~valid] = 0.0

    scale_ratio = np.maximum(
        distances
        / max(median_value, eps),
        eps,
    )
    long_edge_penalty = 1.0 / (
        1.0 + np.power(scale_ratio, 1.35)
    )
    long_edge_penalty[~valid] = 0.0

    density_factor = (
        0.72
        + 0.28 * endpoint_density
    )
    concentration_factor = (
        0.82
        + 0.36 * endpoint_concentration
    )

    score = (
        np.maximum(mutual_transition, eps) ** 1.30
        * np.maximum(short_signal, eps) ** 1.10
        * np.maximum(opportunity, eps) ** 0.72
        * np.maximum(global_rank, eps) ** 0.48
        * np.maximum(long_edge_penalty, eps) ** 0.62
        * density_factor
        * concentration_factor
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score_max = float(np.max(score))
    if not np.isfinite(score_max) or score_max <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= score_max

    row_sum = score.sum(axis=1, keepdims=True)
    row_normalized = score / np.maximum(row_sum, eps)
    row_normalized = 0.5 * (
        row_normalized + row_normalized.T
    )

    normalized_max = float(np.max(row_normalized))
    if np.isfinite(normalized_max) and normalized_max > 0.0:
        row_normalized /= normalized_max

    result = (
        0.78 * score
        + 0.22 * row_normalized
    )

    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    result_max = float(np.max(result))
    if np.isfinite(result_max) and result_max > 0.0:
        result /= result_max

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

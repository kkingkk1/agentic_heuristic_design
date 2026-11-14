import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-diffusion edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & np.isfinite(raw.T)
        & (raw.T >= 0.0)
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    distances = np.maximum(distances, 0.0)

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        scale = max(mad, 0.35 * median, eps)
    else:
        scale = 1.0

    # Robust percentile coordinates avoid letting a few extreme edges dominate.
    sorted_values = np.sort(values)
    ranks_global = np.searchsorted(
        sorted_values,
        values,
        side="right",
    ).astype(float)
    global_percentile = ranks_global / max(float(values.size), 1.0)

    global_signal = np.zeros((n, n), dtype=float)
    global_signal[valid] = 1.0 - global_percentile

    # Each row receives a smooth distribution over its distance ranks.
    masked = np.where(valid, distances, np.inf)
    row_order = np.argsort(masked, axis=1, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    valid_count = np.maximum(valid.sum(axis=1), 1)
    rank_width = np.maximum(
        0.75,
        0.55 * np.sqrt(valid_count.astype(float)),
    )

    rank_kernel = np.exp(
        -np.power(row_rank / rank_width[:, None], 1.35)
    )
    rank_kernel[~valid] = 0.0

    row_mass = rank_kernel.sum(axis=1, keepdims=True)
    local_probability = rank_kernel / np.maximum(row_mass, eps)

    reciprocal_rank = np.sqrt(
        np.maximum(
            local_probability * local_probability.T,
            0.0,
        )
    )
    reciprocal_rank[~valid] = 0.0

    # Node scarcity favors edges incident to nodes with fewer attractive choices.
    degree_mass = np.maximum(local_probability.sum(axis=1), eps)
    scarcity = 1.0 / np.sqrt(degree_mass)
    scarcity /= max(float(np.max(scarcity)), eps)

    scarcity_signal = np.sqrt(
        np.maximum(
            scarcity[:, None] * scarcity[None, :],
            0.0,
        )
    )
    scarcity_signal[~valid] = 0.0

    # Robust distance signal based on a logistic transform around the median.
    distance_median = float(np.median(values))
    centered = (distance_median - distances) / scale
    distance_signal = np.zeros((n, n), dtype=float)
    distance_signal[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -24.0, 24.0))
    )

    # Build a row-stochastic graph and measure multi-step structural support.
    graph = np.sqrt(
        np.maximum(
            local_probability * local_probability.T,
            0.0,
        )
    )
    graph[~valid] = 0.0

    graph_mass = graph.sum(axis=1, keepdims=True)
    transition = graph / np.maximum(graph_mass, eps)

    two_step = transition @ transition
    three_step = two_step @ transition

    diffusion_support = np.sqrt(
        np.maximum(
            two_step * two_step.T,
            0.0,
        )
    )
    diffusion_support += 0.5 * np.sqrt(
        np.maximum(
            three_step * three_step.T,
            0.0,
        )
    )
    diffusion_support[~valid] = 0.0

    support_max = np.max(diffusion_support)
    if np.isfinite(support_max) and support_max > 0.0:
        diffusion_support /= support_max

    # Geometric fusion limits domination by any single signal.
    score = (
        np.maximum(distance_signal, eps) ** 0.27
        * np.maximum(reciprocal_rank, eps) ** 0.31
        * np.maximum(scarcity_signal, eps) ** 0.17
        * np.maximum(diffusion_support + 0.08, eps) ** 0.25
        * np.maximum(0.35 + 0.65 * global_signal, eps) ** 0.10
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    result = score / maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

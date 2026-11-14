import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded edge priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
    )
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    values = raw[valid]
    positive = values[values > 0.0]

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

    eps = np.finfo(float).eps
    scale = max(float(scale), eps)

    upper = max(
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.75)),
        scale,
    )
    replacement = upper + 10.0 * scale

    distances = np.where(valid, raw, replacement)
    np.fill_diagonal(distances, replacement)

    normalized = np.clip(distances / scale, 0.0, 60.0)

    temperature = max(
        0.75 * scale,
        0.10 * float(np.median(values)),
        eps,
    )

    distance_affinity = np.zeros((n, n), dtype=float)
    distance_affinity[valid] = np.exp(
        -normalized[valid] * scale / temperature
    )

    row_mass = distance_affinity.sum(axis=1)
    col_mass = distance_affinity.sum(axis=0)

    row_preference = distance_affinity / np.maximum(
        row_mass[:, None],
        eps,
    )
    col_preference = distance_affinity / np.maximum(
        col_mass[None, :],
        eps,
    )

    row_preference[~valid] = 0.0
    col_preference[~valid] = 0.0

    mutual_preference = np.sqrt(
        np.maximum(
            row_preference * col_preference,
            0.0,
        )
    )
    mutual_preference[~valid] = 0.0

    row_order = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(
        distances,
        axis=0,
        kind="stable",
    )
    col_rank = np.empty((n, n), dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(
        1.0,
        0.85 * np.sqrt(float(n)),
    )

    row_rank_score = np.exp(
        -row_rank / rank_scale
    )
    col_rank_score = np.exp(
        -col_rank / rank_scale
    )

    local_rank = 0.5 * (
        row_rank_score + col_rank_score
    )
    local_rank[~valid] = 0.0

    row_sorted = np.sort(distances, axis=1)
    col_sorted = np.sort(distances, axis=0)

    row_second = row_sorted[:, min(1, n - 1)]
    col_second = col_sorted[min(1, n - 1), :]

    row_gap_scale = np.maximum(
        row_second - row_sorted[:, 0],
        0.15 * scale,
    )
    col_gap_scale = np.maximum(
        col_second - col_sorted[0, :],
        0.15 * scale,
    )

    row_gap = np.maximum(
        row_second[:, None] - distances,
        0.0,
    )
    col_gap = np.maximum(
        col_second[None, :] - distances,
        0.0,
    )

    row_competition = 1.0 - np.exp(
        -row_gap / row_gap_scale[:, None]
    )
    col_competition = 1.0 - np.exp(
        -col_gap / col_gap_scale[None, :]
    )

    competition = 0.5 * (
        row_competition + col_competition
    )
    competition[~valid] = 0.0

    global_rank = np.zeros((n, n), dtype=float)
    flat_values = values.copy()
    order = np.argsort(flat_values, kind="stable")
    flat_rank = np.empty(order.size, dtype=float)
    flat_rank[order] = np.arange(order.size, dtype=float)

    if order.size > 1:
        flat_rank = 1.0 - flat_rank / float(order.size - 1)
    else:
        flat_rank.fill(1.0)

    global_rank[valid] = 0.20 + 0.80 * flat_rank

    structural_signal = (
        0.52 * mutual_preference
        + 0.28 * competition
        + 0.20 * local_rank
    )
    structural_signal[~valid] = 0.0

    directional_signal = (
        0.52 * row_preference
        + 0.28 * col_preference
        + 0.20 * local_rank
    )
    directional_signal[~valid] = 0.0

    core = (
        0.42 * distance_affinity
        + 0.22 * directional_signal
        + 0.18 * structural_signal
        + 0.12 * global_rank
        + 0.06 * competition
    )

    core[~valid] = 0.0
    core = np.clip(core, 0.0, None)

    maximum = float(np.max(core))
    if not np.isfinite(maximum) or maximum <= 0.0:
        core[valid] = 1.0
        maximum = 1.0

    core /= maximum

    exploration_floor = 0.035
    result[valid] = (
        exploration_floor
        + (1.0 - exploration_floor) * core[valid]
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

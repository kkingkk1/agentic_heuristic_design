import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded edge priors for stochastic TSP solution sampling."""
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
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

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

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    finite_values = values[np.isfinite(values)]
    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    temperature = max(mad, 0.10 * safe_scale, eps)

    distances = np.where(valid, raw, np.inf)
    np.fill_diagonal(distances, np.inf)

    def row_max_normalize(feature: np.ndarray) -> np.ndarray:
        result = np.zeros_like(feature, dtype=float)
        maxima = np.max(
            np.where(valid, feature, 0.0),
            axis=1,
            keepdims=True,
        )
        result[valid] = (
            feature[valid]
            / np.maximum(maxima, eps)[
                np.broadcast_to(
                    np.arange(n)[:, None],
                    (n, n),
                )[valid]
            ]
        )
        return np.clip(result, 0.0, 1.0)

    distance_affinity = np.zeros((n, n), dtype=float)
    distance_affinity[valid] = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (raw[valid] - median) / temperature,
                -40.0,
                40.0,
            )
        )
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        raw[valid] + 0.04 * safe_scale
    )

    inverse = row_max_normalize(inverse)
    distance_affinity = row_max_normalize(distance_affinity)

    order = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, 0.70 * np.sqrt(float(n)))
    rank_signal = np.zeros((n, n), dtype=float)
    rank_signal[valid] = np.exp(
        -ranks[valid] / rank_scale
    )
    rank_signal = row_max_normalize(rank_signal)

    incoming_order = np.argsort(
        distances,
        axis=0,
        kind="stable",
    )
    incoming_ranks = np.empty((n, n), dtype=float)
    incoming_ranks[
        incoming_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    incoming_signal = np.zeros((n, n), dtype=float)
    incoming_signal[valid] = np.exp(
        -incoming_ranks[valid] / rank_scale
    )
    incoming_signal = np.clip(incoming_signal, 0.0, 1.0)

    local_competition = np.sqrt(
        np.maximum(
            rank_signal * incoming_signal,
            0.0,
        )
    )

    sorted_rows = np.sort(
        np.where(valid, raw, np.inf),
        axis=1,
    )
    best = sorted_rows[:, 0]
    second = sorted_rows[:, min(1, n - 1)]
    gap_scale = np.maximum(
        second - best,
        0.12 * safe_scale,
    )

    regret = np.zeros((n, n), dtype=float)
    regret[valid] = 1.0 - np.exp(
        -np.maximum(
            second[:, None] - raw,
            0.0,
        )[valid]
        / gap_scale[:, None][valid]
    )
    regret = np.clip(regret, 0.0, 1.0)

    bypass = np.full(
        (n, n),
        np.inf,
        dtype=float,
    )

    for intermediate in range(n):
        left_valid = valid[:, intermediate][:, None]
        right_valid = valid[intermediate, :][None, :]
        candidate = (
            distances[:, intermediate][:, None]
            + distances[intermediate, :][None, :]
        )
        candidate_valid = left_valid & right_valid
        bypass = np.minimum(
            bypass,
            np.where(candidate_valid, candidate, np.inf),
        )

    detour = np.zeros((n, n), dtype=float)
    usable_detour = valid & np.isfinite(bypass)
    detour[usable_detour] = 1.0 - np.exp(
        -np.maximum(
            bypass[usable_detour] - raw[usable_detour],
            0.0,
        )
        / max(0.65 * safe_scale, eps)
    )
    detour = np.clip(detour, 0.0, 1.0)

    global_rank = np.zeros((n, n), dtype=float)
    sorted_values = np.sort(values)
    percentile = (
        np.searchsorted(
            sorted_values,
            values,
            side="right",
        ).astype(float)
        / max(float(values.size), 1.0)
    )
    global_rank[valid] = 1.0 - percentile

    structural_signal = np.clip(
        0.65 * regret
        + 0.35 * detour,
        0.0,
        1.0,
    )

    incoming_balance = np.zeros((n, n), dtype=float)
    incoming_mass = np.sum(
        np.where(valid, distance_affinity, 0.0),
        axis=0,
        keepdims=True,
    )
    incoming_balance[valid] = (
        distance_affinity[valid]
        / np.maximum(incoming_mass, eps)[
            np.broadcast_to(
                np.arange(n)[None, :],
                (n, n),
            )[valid]
        ]
    )
    incoming_balance = np.clip(
        incoming_balance * float(max(n - 1, 1)),
        0.0,
        1.0,
    )

    diversity_signal = np.clip(
        0.60 * local_competition
        + 0.25 * incoming_balance
        + 0.15 * global_rank,
        0.0,
        1.0,
    )

    score = (
        0.43 * distance_affinity
        + 0.25 * diversity_signal
        + 0.17 * structural_signal
        + 0.10 * inverse
        + 0.05 * global_rank
    )

    score[~valid] = 0.0

    valid_count = np.maximum(
        valid.sum(axis=1, keepdims=True),
        1,
    )
    row_floor = 0.025 / valid_count
    score = np.where(
        valid,
        score + row_floor,
        0.0,
    )

    row_mass = score.sum(axis=1, keepdims=True)
    row_balanced = score / np.maximum(row_mass, eps)

    result = (
        0.78 * score
        + 0.22 * row_balanced
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        maximum = float(np.max(result))

    if maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)

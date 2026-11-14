import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, non-negative edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid | directed_valid.T

    if not np.any(valid):
        return result

    symmetric = np.full((n, n), np.inf, dtype=float)

    both = directed_valid & directed_valid.T
    only_forward = directed_valid & ~directed_valid.T
    only_reverse = directed_valid.T & ~directed_valid

    symmetric[both] = 0.5 * (
        raw[both] + raw.T[both]
    )
    symmetric[only_forward] = raw[only_forward]
    symmetric[only_reverse] = raw.T[only_reverse]
    np.fill_diagonal(symmetric, np.inf)

    values = symmetric[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        median = float(np.median(positive))
        q25, q75 = np.quantile(positive, [0.25, 0.75])
        iqr = float(q75 - q25)
        mad = float(np.median(np.abs(positive - median)))

        scale = max(
            iqr / 1.349,
            mad / 0.6745,
            0.10 * max(median, 1.0),
            eps,
        )
        spread_ratio = float(
            np.std(positive)
            / max(abs(median), scale, eps)
        )
    else:
        median = 0.0
        scale = 1.0
        spread_ratio = 0.0

    maximum = float(np.max(values))
    replacement = maximum + 6.0 * scale

    distances = np.where(valid, symmetric, replacement)
    np.fill_diagonal(distances, replacement)

    scaled = np.clip(distances / max(scale, eps), 0.0, 60.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-scaled[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        distances[valid] + 0.08 * max(scale, eps)
    )

    inverse_max = float(np.max(inverse))
    if inverse_max > 0.0:
        inverse /= inverse_max

    distance_signal = 0.78 * affinity + 0.22 * inverse
    distance_signal[~valid] = 0.0

    row_mass = distance_signal.sum(axis=1)
    col_mass = distance_signal.sum(axis=0)

    row_preference = distance_signal / np.maximum(
        row_mass[:, None],
        eps,
    )
    col_preference = distance_signal / np.maximum(
        col_mass[None, :],
        eps,
    )

    mutual_preference = np.sqrt(
        np.maximum(row_preference * col_preference, 0.0)
    )
    mutual_preference[~valid] = 0.0

    order = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )
    rank = np.empty((n, n), dtype=float)
    rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_signal = np.exp(-rank / rank_scale)
    rank_signal[~valid] = 0.0

    sorted_distances = np.sort(distances, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, min(1, n - 1)]
    second = np.where(
        np.isfinite(second),
        second,
        best + scale,
    )

    gap_scale = np.maximum(
        second - best,
        0.20 * max(scale, eps),
    )

    local_gap = np.maximum(
        second[:, None] - distances,
        0.0,
    )
    local_competition = 1.0 - np.exp(
        -local_gap / gap_scale[:, None]
    )
    local_competition[~valid] = 0.0

    # Bidirectional regret: preserve edges whose replacement is substantially
    # worse at either endpoint.
    row_regret = np.maximum(second[:, None] - distances, 0.0)
    row_regret = 1.0 - np.exp(
        -row_regret / np.maximum(gap_scale[:, None], eps)
    )

    sorted_columns = np.sort(distances, axis=0)
    best_column = sorted_columns[0, :]
    second_column = sorted_columns[min(1, n - 1), :]
    second_column = np.where(
        np.isfinite(second_column),
        second_column,
        best_column + scale,
    )

    column_gap_scale = np.maximum(
        second_column - best_column,
        0.20 * max(scale, eps),
    )
    column_regret = np.maximum(
        second_column[None, :] - distances,
        0.0,
    )
    column_regret = 1.0 - np.exp(
        -column_regret / np.maximum(column_gap_scale[None, :], eps)
    )

    regret = 0.5 * (row_regret + column_regret)
    regret[~valid] = 0.0

    # Two-hop detour signal rewards edges that are difficult to bypass.
    bypass = np.full((n, n), replacement, dtype=float)
    for middle in range(n):
        candidate = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = replacement
    detour = np.maximum(bypass - distances, 0.0)
    detour_signal = 1.0 - np.exp(
        -detour / max(0.75 * scale, eps)
    )
    detour_signal[~valid] = 0.0

    variation = float(np.clip(spread_ratio, 0.0, 2.0))
    distance_weight = 0.37 + 0.07 * min(variation, 1.0)
    mutual_weight = 0.25 - 0.025 * min(variation, 1.0)
    competition_weight = 0.17
    rank_weight = 0.11
    regret_weight = 0.07
    detour_weight = 1.0 - (
        distance_weight
        + mutual_weight
        + competition_weight
        + rank_weight
        + regret_weight
    )

    score = (
        distance_weight * distance_signal
        + mutual_weight * mutual_preference
        + competition_weight * local_competition
        + rank_weight * rank_signal
        + regret_weight * regret
        + detour_weight * detour_signal
    )
    score[~valid] = 0.0

    positive_scores = score[valid]
    if positive_scores.size:
        score_max = float(np.max(positive_scores))
        floor = 0.025 * score_max
        score[valid] = floor + 0.975 * score[valid]

    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    score[~valid] = 0.0
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

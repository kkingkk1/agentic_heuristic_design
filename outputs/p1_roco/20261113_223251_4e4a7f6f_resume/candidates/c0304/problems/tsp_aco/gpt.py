import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, non-negative edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid | directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

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

    if positive.size:
        median = float(np.median(positive))
        q25, q75 = np.quantile(positive, [0.25, 0.75])
        iqr = float(q75 - q25)
        mad = float(np.median(np.abs(positive - median)))

        scale = max(
            iqr / 1.349,
            mad / 0.6745,
            0.10 * max(median, 1.0),
            np.finfo(float).eps,
        )

        spread_ratio = (
            float(np.std(positive))
            / max(abs(median), scale, np.finfo(float).eps)
        )
    else:
        median = 0.0
        scale = 1.0
        spread_ratio = 0.0

    maximum = float(np.max(values))
    replacement = maximum + 6.0 * scale

    distances = np.where(valid, symmetric, replacement)
    np.fill_diagonal(distances, replacement)

    eps = np.finfo(float).eps

    # Robust distance affinity with a mild inverse-distance supplement.
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

    # Row and column soft preferences capture mutual endpoint compatibility.
    row_mass = distance_signal.sum(axis=1)
    col_mass = distance_signal.sum(axis=0)

    row_preference = (
        distance_signal
        / np.maximum(row_mass[:, None], eps)
    )
    col_preference = (
        distance_signal
        / np.maximum(col_mass[None, :], eps)
    )

    mutual_preference = np.sqrt(
        np.maximum(row_preference * col_preference, 0.0)
    )
    mutual_preference[~valid] = 0.0

    # Stable local rank, excluding diagonal and invalid edges.
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

    # Local competition: reward edges that are clearly better than alternatives
    # at both endpoints, while avoiding a hard zero for merely adequate edges.
    sorted_distances = np.sort(distances, axis=1)

    best = sorted_distances[:, 0]
    second = sorted_distances[:, min(1, n - 1)]
    second = np.where(
        np.isfinite(second),
        second,
        best + scale,
    )

    row_gap_scale = np.maximum(
        second - best,
        0.20 * max(scale, eps),
    )

    row_gap = np.maximum(
        second[:, None] - distances,
        0.0,
    )

    row_competition = 1.0 - np.exp(
        -row_gap / row_gap_scale[:, None]
    )
    row_competition[~valid] = 0.0

    # Symmetric input is used after preprocessing, but averaging both endpoint
    # views keeps the signal well behaved for originally asymmetric inputs.
    local_competition = 0.5 * (
        row_competition + row_competition.T
    )
    local_competition[~valid] = 0.0

    # Adapt weights only mildly. Strong distance variation favors direct
    # distance evidence; nearly uniform distances favor structural signals.
    variation = float(np.clip(spread_ratio, 0.0, 2.0))
    distance_weight = 0.40 + 0.08 * min(variation, 1.0)
    mutual_weight = 0.27 - 0.03 * min(variation, 1.0)
    competition_weight = 0.21
    rank_weight = 1.0 - (
        distance_weight + mutual_weight + competition_weight
    )

    score = (
        distance_weight * distance_signal
        + mutual_weight * mutual_preference
        + competition_weight * local_competition
        + rank_weight * rank_signal
    )
    score[~valid] = 0.0

    # Small exploration floor: preserve every valid edge without making poor
    # edges competitive with clearly better local choices.
    positive_scores = score[valid]
    if positive_scores.size:
        score_max = float(np.max(positive_scores))
        floor = 0.025 * score_max
        score[valid] = floor + (1.0 - 0.025) * score[valid]

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

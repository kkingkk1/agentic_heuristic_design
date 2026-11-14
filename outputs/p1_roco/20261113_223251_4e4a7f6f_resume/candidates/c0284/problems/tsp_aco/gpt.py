import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    if not np.isfinite(maximum):
        maximum = scale

    penalty = maximum + 16.0 * scale
    distances = np.where(valid, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    valid_distances = distances[valid]
    temperature = max(
        0.24 * float(np.median(valid_distances)),
        np.finfo(float).eps,
    )

    shortness = np.exp(-np.minimum(distances, penalty) / temperature)
    shortness[~valid] = 0.0
    np.fill_diagonal(shortness, 0.0)

    peak = float(shortness.max())
    if not np.isfinite(peak) or peak <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    shortness /= peak

    sorted_distances = np.sort(distances, axis=1)
    nearest = sorted_distances[:, 0]
    second_nearest = (
        sorted_distances[:, 1] if n > 2 else nearest
    )

    gap = np.maximum(
        second_nearest - nearest,
        0.14 * scale,
    )
    gap = np.maximum(gap, np.finfo(float).eps)

    preference = np.exp(
        np.clip(
            -(distances - nearest[:, None]) / gap[:, None],
            -24.0,
            0.0,
        )
    )
    preference[~valid] = 0.0
    np.fill_diagonal(preference, 0.0)

    preference /= np.maximum(
        preference.max(axis=1, keepdims=True),
        np.finfo(float).eps,
    )

    endpoint_competition = np.sqrt(
        np.maximum(preference, 0.0)
        * np.maximum(preference.T, 0.0)
    )

    regret = np.exp(
        np.clip(
            (second_nearest[:, None] - distances) / gap[:, None],
            -18.0,
            18.0,
        )
    )
    regret[~valid] = 0.0
    np.fill_diagonal(regret, 0.0)

    endpoint_regret = np.sqrt(
        np.maximum(regret, 0.0)
        * np.maximum(regret.T, 0.0)
    )

    best_two_hop = np.full((n, n), penalty, dtype=float)
    for middle in range(n):
        via_middle = (
            distances[:, middle, None]
            + distances[None, middle, :]
        )
        best_two_hop = np.minimum(best_two_hop, via_middle)

    np.fill_diagonal(best_two_hop, penalty)

    bypass_gain = np.maximum(best_two_hop - distances, 0.0)
    bypass_scale = max(0.68 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-bypass_gain / bypass_scale)
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    score = (
        0.42 * shortness
        + 0.24 * endpoint_competition
        + 0.18 * endpoint_regret
        + 0.16 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(24):
        row_totals = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_totals, np.finfo(float).eps)
        balanced = 0.5 * (balanced + balanced.T)

    balanced_peak = float(balanced.max())
    if np.isfinite(balanced_peak) and balanced_peak > 0.0:
        balanced /= balanced_peak

    result = 0.62 * score + 0.38 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    result_peak = float(result.max())
    if np.isfinite(result_peak) and result_peak > 0.0:
        result /= result_peak

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

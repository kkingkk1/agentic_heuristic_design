import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)

    valid = finite_nonnegative & finite_nonnegative.T
    valid &= ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)
    observed = symmetric[valid]

    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)
    largest = float(np.max(observed)) if observed.size else scale
    penalty = largest + 12.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = symmetric[valid]
    np.fill_diagonal(distances, penalty)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / scale, 50.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + eps)
    inverse /= max(float(np.max(inverse)), eps)

    row_mass = affinity.sum(axis=1, keepdims=True)
    col_mass = affinity.sum(axis=0, keepdims=True)

    row_share = affinity / np.maximum(row_mass, eps)
    col_share = affinity / np.maximum(col_mass, eps)

    competition = np.sqrt(
        np.maximum(row_share * col_share, 0.0)
    )

    sorted_rows = np.sort(
        np.where(valid, distances, penalty),
        axis=1,
    )
    best_row = sorted_rows[:, 0]
    second_row = sorted_rows[:, min(1, n - 1)]

    sorted_cols = np.sort(
        np.where(valid, distances.T, penalty),
        axis=1,
    )
    best_col = sorted_cols[:, 0]
    second_col = sorted_cols[:, min(1, n - 1)]

    del best_row, best_col

    row_exclusivity = np.maximum(
        second_row[:, None] - distances,
        0.0,
    )
    col_exclusivity = np.maximum(
        second_col[None, :] - distances,
        0.0,
    )

    gap_scale = float(
        np.median(
            np.concatenate(
                [
                    np.maximum(second_row - sorted_rows[:, 0], 0.0),
                    np.maximum(second_col - sorted_cols[:, 0], 0.0),
                ]
            )
        )
    )
    gap_scale = max(gap_scale, 0.18 * scale, eps)

    exclusivity = 1.0 - np.exp(
        -0.65 * (row_exclusivity + col_exclusivity) / gap_scale
    )
    exclusivity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    for intermediate in range(n):
        candidate = (
            distances[:, intermediate, None]
            + distances[None, intermediate, :]
        )
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, penalty)

    bypass_advantage = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -0.85 * bypass_advantage / max(0.75 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    log_affinity = np.log(np.maximum(inverse, eps))
    log_competition = np.log(np.maximum(competition, eps))
    log_exclusivity = np.log(np.maximum(exclusivity, eps))
    log_irreplaceability = np.log(
        np.maximum(irreplaceability, eps)
    )

    score_log = (
        0.42 * log_affinity
        + 0.27 * log_competition
        + 0.17 * log_exclusivity
        + 0.14 * log_irreplaceability
    )

    valid_log_distance = np.log1p(
        distances[valid] / max(scale, eps)
    )
    median_log = float(np.median(valid_log_distance))
    dispersion = float(
        np.median(np.abs(valid_log_distance - median_log))
    )

    temperature = float(
        np.clip(0.82 + 0.35 * dispersion, 0.82, 1.28)
    )

    weights = np.zeros((n, n), dtype=float)
    masked_scores = score_log[valid] / temperature
    offset = float(np.max(masked_scores))

    weights[valid] = np.exp(
        np.clip(masked_scores - offset, -60.0, 0.0)
    )

    exploration = float(
        np.clip(0.06 + 0.035 * dispersion, 0.06, 0.12)
    )
    valid_count = float(np.sum(valid))

    weights = (
        (1.0 - exploration) * weights
        + exploration * valid.astype(float) / valid_count
    )
    weights = 0.5 * (weights + weights.T)

    for _ in range(16):
        row_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.maximum(row_sum, eps)
        weights = 0.5 * (weights + weights.T)
        weights[~valid] = 0.0

    structural = np.sqrt(
        np.maximum(inverse * affinity, 0.0)
    )
    structural[~valid] = 0.0

    result = 0.70 * weights + 0.30 * structural
    result = 0.5 * (result + result.T)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

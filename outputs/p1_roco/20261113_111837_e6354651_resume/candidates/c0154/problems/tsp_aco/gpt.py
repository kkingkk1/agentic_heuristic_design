import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge priorities for stochastic TSP solution sampling."""
    matrix = np.asarray(distance_matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(matrix) & (matrix >= 0.0)
    finite &= ~np.eye(n, dtype=bool)

    if not np.any(finite):
        return result

    undirected = np.full((n, n), np.inf, dtype=float)
    forward = np.where(finite, matrix, np.inf)
    reverse = forward.T

    available = np.isfinite(forward) | np.isfinite(reverse)
    undirected[available] = np.minimum(forward[available], reverse[available])
    np.fill_diagonal(undirected, np.inf)

    valid = np.isfinite(undirected)
    if not np.any(valid):
        return result

    positive = undirected[valid & (undirected > 0.0)]
    if positive.size:
        reference = float(np.percentile(positive, 50.0))
    else:
        reference = 1.0
    reference = max(reference, np.finfo(float).eps)

    row_values = np.where(valid, undirected, np.nan)
    row_median = np.nanmedian(row_values, axis=1)
    row_median = np.where(
        np.isfinite(row_median),
        row_median,
        reference,
    )
    row_median = np.maximum(row_median, np.finfo(float).eps)

    centered = undirected / row_median[:, None]
    centered = np.where(valid, centered, np.inf)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_advantage = 1.0 / (1.0 + np.maximum(centered - 1.0, 0.0))
        relative_advantage = np.where(valid, relative_advantage, 0.0)

        preference_logits = -np.minimum(centered, 30.0)
        preference_logits = np.where(valid, preference_logits, -np.inf)
        row_max = np.max(preference_logits, axis=1, keepdims=True)
        row_weights = np.exp(preference_logits - row_max)
        row_weights = np.where(valid, row_weights, 0.0)
        row_totals = np.sum(row_weights, axis=1, keepdims=True)
        preferences = np.divide(
            row_weights,
            row_totals,
            out=np.zeros_like(row_weights),
            where=row_totals > 0.0,
        )

    endpoint_support = np.maximum(preferences, preferences.T)
    pair_support = np.sqrt(preferences * preferences.T)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(
            preferences > 0.0,
            -preferences * np.log(preferences),
            0.0,
        )
    entropy = np.sum(entropy_terms, axis=1)
    max_entropy = np.log(np.maximum(np.sum(valid, axis=1), 2))
    concentration = 1.0 - np.divide(
        entropy,
        max_entropy,
        out=np.zeros_like(entropy),
        where=max_entropy > 0.0,
    )
    concentration = np.clip(concentration, 0.0, 1.0)

    endpoint_concentration = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    normalized_global = undirected / reference
    global_quality = 1.0 / (
        1.0 + np.maximum(normalized_global, 0.0)
    )
    global_quality = np.where(valid, global_quality, 0.0)

    inverse_distance = np.where(
        valid,
        1.0 / np.maximum(undirected, np.finfo(float).eps),
        0.0,
    )
    inverse_distance /= np.maximum(
        np.max(inverse_distance, axis=1, keepdims=True),
        np.finfo(float).eps,
    )

    closure = inverse_distance @ inverse_distance.T
    closure = np.sqrt(np.maximum(closure, 0.0))
    closure_max = np.max(closure, initial=0.0)
    if closure_max > 0.0:
        closure /= closure_max
    closure = np.where(valid, closure, 0.0)

    score = (
        0.35 * relative_advantage
        + 0.25 * endpoint_support
        + 0.18 * pair_support
        + 0.12 * global_quality
        + 0.10 * closure
    )

    score *= 0.85 + 0.15 * endpoint_concentration
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score, initial=0.0))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priorities based on directness and two-hop indispensability."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, distances, np.inf)

    finite_values = work[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.max(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    safe_work = np.where(np.isfinite(work), work, 0.0)

    outgoing_mass = np.sum(
        np.exp(-np.minimum(safe_work / scale, 60.0)) * valid,
        axis=1,
    )
    incoming_mass = np.sum(
        np.exp(-np.minimum(safe_work / scale, 60.0)) * valid,
        axis=0,
    )

    endpoint_support = 2.0 * outgoing_mass[:, None] * incoming_mass[None, :]
    endpoint_support = endpoint_support / (
        outgoing_mass[:, None] + incoming_mass[None:] + eps
    )
    endpoint_factor = 0.75 + 0.25 * np.tanh(endpoint_support)

    normalized_distance = np.divide(
        work,
        scale,
        out=np.full_like(work, np.inf),
        where=valid,
    )

    direct_attraction = np.exp(
        -np.minimum(normalized_distance, 60.0)
    )

    if n > 2:
        two_hop = work[:, :, None] + work[None, :, :]
        intermediate_valid = (
            valid[:, :, None]
            & valid[None, :, :]
        )

        two_hop = np.where(
            intermediate_valid,
            two_hop,
            np.inf,
        )

        np.fill_diagonal(two_hop[:, :, :], np.inf)
        best_alternative = np.min(two_hop, axis=1)

        usable_alternative = np.isfinite(best_alternative)
        safe_alternative = np.where(
            usable_alternative,
            best_alternative,
            work,
        )

        direct_safe = np.maximum(work, eps)
        replacement_ratio = np.divide(
            safe_alternative,
            direct_safe,
            out=np.ones_like(work),
            where=valid,
        )

        indispensability = np.tanh(
            np.maximum(np.log1p(replacement_ratio), 0.0)
        )
        indispensability = np.where(valid, indispensability, 0.0)
    else:
        indispensability = valid.astype(float)

    row_mass = outgoing_mass[:, None]
    col_mass = incoming_mass[None, :]
    balance = np.divide(
        2.0 * np.minimum(row_mass, col_mass),
        row_mass + col_mass + eps,
    )
    balance_factor = 0.90 + 0.10 * balance

    score = (
        direct_attraction
        * (0.55 + 0.45 * indispensability)
        * endpoint_factor
        * balance_factor
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score

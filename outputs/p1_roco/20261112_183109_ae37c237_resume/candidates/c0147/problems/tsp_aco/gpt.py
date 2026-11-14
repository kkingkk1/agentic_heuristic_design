import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priorities using global ranks and triangle-detour savings."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances > 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    work = np.where(valid, distances, np.inf)
    np.fill_diagonal(work, np.inf)

    order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    reverse_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[reverse_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    rank_denominator = float(max(1, n - 1))
    outgoing_preference = 1.0 - row_rank / rank_denominator
    incoming_preference = 1.0 - col_rank / rank_denominator

    outgoing_preference = np.clip(outgoing_preference, 0.0, 1.0)
    incoming_preference = np.clip(incoming_preference, 0.0, 1.0)

    rank_score = np.sqrt(outgoing_preference * incoming_preference)

    two_hop = np.full((n, n), np.inf, dtype=float)
    for intermediate in range(n):
        candidate = (
            work[:, intermediate, None]
            + work[intermediate, None, :]
        )
        two_hop = np.minimum(two_hop, candidate)

    savings = two_hop - work
    finite_savings = savings[valid & np.isfinite(savings)]

    if finite_savings.size:
        low = float(np.quantile(finite_savings, 0.10))
        high = float(np.quantile(finite_savings, 0.90))
        span = max(high - low, np.finfo(float).eps)
        bridge_score = np.clip((savings - low) / span, 0.0, 1.0)
    else:
        bridge_score = np.zeros((n, n), dtype=float)

    bridge_score = np.nan_to_num(
        bridge_score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    finite_distances = work[valid]
    distance_low = float(np.quantile(finite_distances, 0.10))
    distance_high = float(np.quantile(finite_distances, 0.90))
    distance_span = max(
        distance_high - distance_low,
        np.finfo(float).eps,
    )

    global_shortness = 1.0 - np.clip(
        (work - distance_low) / distance_span,
        0.0,
        1.0,
    )

    accessibility = 0.5 * (
        outgoing_preference + incoming_preference
    )

    score = (
        0.45 * rank_score
        + 0.35 * bridge_score
        + 0.20 * accessibility
    )

    score *= 0.65 + 0.35 * global_shortness
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)

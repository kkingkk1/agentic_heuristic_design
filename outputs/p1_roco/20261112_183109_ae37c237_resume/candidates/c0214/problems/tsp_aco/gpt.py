import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directed edge priorities for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    elif values.size:
        scale = float(np.mean(values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    normalized = np.divide(
        d,
        scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    # Heavy-tailed global affinity: zero and short distances remain distinguishable
    # without exponential underflow or local-neighbor normalization.
    affinity = np.zeros_like(d, dtype=float)
    affinity[valid] = 1.0 / (
        1.0 + np.square(np.clip(normalized[valid], 0.0, 1e150))
    )

    outgoing_mass = affinity.sum(axis=1)
    incoming_mass = affinity.sum(axis=0)

    safe_outgoing = np.maximum(outgoing_mass, eps)
    safe_incoming = np.maximum(incoming_mass, eps)

    # A destination is preferred when it still offers a strong continuation.
    continuation = np.zeros(n, dtype=float)
    for j in range(n):
        exits = affinity[j].copy()
        exits[j] = 0.0
        continuation[j] = np.max(exits)

    # A source is preferred when it has a strong feasible predecessor.
    predecessor = np.zeros(n, dtype=float)
    for i in range(n):
        entries = affinity[:, i].copy()
        entries[i] = 0.0
        predecessor[i] = np.max(entries)

    max_continuation = max(float(np.max(continuation)), eps)
    max_predecessor = max(float(np.max(predecessor)), eps)

    continuation = continuation / max_continuation
    predecessor = predecessor / max_predecessor

    # Mild degree pressure avoids concentrating all probability on highly
    # attractive hubs while retaining exploration.
    source_factor = 1.0 / np.sqrt(safe_outgoing)
    target_factor = 1.0 / np.sqrt(safe_incoming)

    source_factor /= max(float(np.max(source_factor)), eps)
    target_factor /= max(float(np.max(target_factor)), eps)

    score = affinity.copy()
    score *= 0.70 + 0.30 * continuation[None, :]
    score *= 0.70 + 0.30 * predecessor[:, None]
    score *= 0.85 + 0.15 * source_factor[:, None]
    score *= 0.85 + 0.15 * target_factor[None, :]

    score[~valid] = 0.0
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )

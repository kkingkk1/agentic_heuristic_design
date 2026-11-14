import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a directed random-walk and bypass-criticality prior for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0.0):
        raise ValueError("distance_matrix must be nonnegative")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    temperature = max(0.55 * scale, np.finfo(float).eps)

    logits = -work / temperature
    logits = np.clip(logits, -60.0, 0.0)
    transition = np.exp(logits)
    transition[~valid] = 0.0

    row_mass = transition.sum(axis=1, keepdims=True)
    transition = np.divide(
        transition,
        row_mass,
        out=np.zeros_like(transition),
        where=row_mass > 0.0,
    )

    stationary = np.full(n, 1.0 / n, dtype=float)
    for _ in range(80):
        updated = stationary @ transition
        total = float(updated.sum())
        if total > 0.0 and np.isfinite(total):
            updated /= total
        else:
            updated[:] = 1.0 / n
        if np.max(np.abs(updated - stationary)) < 1e-10:
            stationary = updated
            break
        stationary = updated

    flow = stationary[:, None] * transition

    if n > 2:
        two_step = np.full((n, n), np.inf, dtype=float)

        for k in range(n):
            left = work[:, k][:, None]
            right = work[k, :][None, :]
            candidate = left + right
            candidate[k, :] = np.inf
            candidate[:, k] = np.inf
            two_step = np.minimum(two_step, candidate)

        bypass_margin = two_step - work
        bypass_margin = np.maximum(bypass_margin, 0.0)
        criticality = 1.0 - np.exp(
            -bypass_margin / max(scale, np.finfo(float).eps)
        )
        criticality[~valid] = 0.0
    else:
        criticality = np.zeros((n, n), dtype=float)

    return_cost = work.T
    returnability = np.exp(
        -np.minimum((work + return_cost) / (2.0 * scale), 40.0)
    )
    returnability[~valid] = 0.0

    score = flow * (
        0.55
        + 0.85 * criticality
        + 0.20 * returnability
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    mean_score = float(score[valid].mean()) if np.any(valid) else 0.0
    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

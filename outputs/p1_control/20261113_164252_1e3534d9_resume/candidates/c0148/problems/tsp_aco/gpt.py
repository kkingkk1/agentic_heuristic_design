import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return cycle-compatible stationary-flow priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        temperature = max(float(np.median(positive)) * 0.45, eps)
    else:
        finite = d[valid]
        temperature = max(float(np.median(finite)) if finite.size else 1.0, eps)

    safe = np.where(valid, d, np.inf)
    row_min = np.min(safe, axis=1)
    active_rows = np.isfinite(row_min)

    shifted = np.zeros_like(d, dtype=float)
    np.subtract(
        d,
        row_min[:, None],
        out=shifted,
        where=valid,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        transition = np.exp(-np.minimum(shifted / temperature, 60.0))

    transition[~valid] = 0.0

    row_mass = transition.sum(axis=1, keepdims=True)
    transition = np.divide(
        transition,
        row_mass,
        out=np.zeros_like(transition),
        where=row_mass > 0.0,
    )

    active = active_rows.astype(float)
    active_count = float(np.sum(active))

    if active_count == 0.0:
        return result

    teleport = active / active_count
    stationary = teleport.copy()

    for _ in range(128):
        updated = 0.86 * (stationary @ transition) + 0.14 * teleport
        if np.max(np.abs(updated - stationary)) < 1e-12:
            stationary = updated
            break
        stationary = updated

    flow = stationary[:, None] * transition

    two_step = transition @ transition
    one_step_return = transition.T
    two_step_return = two_step.T

    closure = 0.35 * one_step_return + 0.65 * two_step_return
    closure[~valid] = 0.0

    closure_max = float(np.max(closure))
    if closure_max > 0.0:
        closure /= closure_max

    score = flow * (0.15 + closure)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

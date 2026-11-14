import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d > 0)
    finite &= finite.T

    work = np.empty_like(d, dtype=float)
    np.add(d, d.T, out=work)
    work *= 0.5
    work[~finite] = np.inf
    np.fill_diagonal(work, np.inf)

    valid = work[np.isfinite(work)]
    if valid.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(valid))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(valid))
    scale = max(scale, np.finfo(float).eps)

    proximity = np.exp(-work / scale)
    proximity[~finite] = 0.0
    np.fill_diagonal(proximity, 0.0)

    row_mass = proximity.sum(axis=1)
    safe_mass = np.maximum(row_mass, np.finfo(float).eps)

    scarcity = 1.0 / safe_mass
    scarcity /= max(float(scarcity.mean()), np.finfo(float).eps)

    local_strength = proximity / safe_mass[:, None]
    mutual_support = np.sqrt(local_strength * local_strength.T)

    score = np.zeros_like(work)
    np.divide(1.0, work, out=score, where=finite)

    score *= 0.55 + 0.30 * np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    score *= 0.70 + 0.60 * mutual_support
    score *= 0.75 + 0.25 * proximity
    score[~finite] = 0.0

    score += score.T
    score *= 0.5
    np.fill_diagonal(score, 0.0)

    maximum = score.max(initial=0.0)
    if maximum > 0.0:
        score /= maximum

    return score

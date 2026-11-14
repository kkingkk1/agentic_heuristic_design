import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priors based on direct-edge detour advantage."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, d, np.inf)
    score = np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue

            detour_values = []
            direct = safe[i, j]

            for k in range(n):
                if k == i or k == j:
                    continue
                if valid[i, k] and valid[k, j]:
                    detour = safe[i, k] + safe[k, j]
                    if np.isfinite(detour):
                        detour_values.append(detour)

            if not detour_values:
                score[i, j] = 0.0
                continue

            detours = np.asarray(detour_values, dtype=float)
            best_detour = float(np.min(detours))

            # Scale-free direct-versus-detour advantage.
            advantage = (best_detour - direct) / (
                best_detour + direct + eps
            )
            advantage = float(np.clip(advantage, -1.0, 1.0))

            # Fraction of available intermediate nodes for which the edge
            # is shorter than the corresponding two-edge detour.
            dominance = float(np.mean(detours > direct))

            # Suppress edges that do not provide a meaningful shortcut.
            score[i, j] = max(0.0, advantage) * (0.5 + 0.5 * dominance)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

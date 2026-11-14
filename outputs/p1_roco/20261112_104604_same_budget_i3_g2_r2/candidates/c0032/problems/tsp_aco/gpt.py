import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, distances, np.inf)
    eps = np.finfo(float).eps

    best_outgoing = np.min(work, axis=1, keepdims=True)
    best_incoming = np.min(work, axis=0, keepdims=True)

    relative_outgoing = best_outgoing / (work + eps)
    relative_incoming = best_incoming / (work + eps)

    locality = np.sqrt(
        np.clip(relative_outgoing, 0.0, 1.0)
        * np.clip(relative_incoming, 0.0, 1.0)
    )

    if n >= 3:
        second_outgoing = np.partition(work, 1, axis=1)[:, 1:2]
        second_incoming = np.partition(work, 1, axis=0)[1:2, :]
    else:
        second_outgoing = work
        second_incoming = work

    outgoing_regret = (second_outgoing - work) / (second_outgoing + eps)
    incoming_regret = (second_incoming - work) / (second_incoming + eps)

    outgoing_regret = np.where(np.isfinite(outgoing_regret), outgoing_regret, 1.0)
    incoming_regret = np.where(np.isfinite(incoming_regret), incoming_regret, 1.0)

    regret = 0.5 * (
        np.clip(outgoing_regret, 0.0, 1.0)
        + np.clip(incoming_regret, 0.0, 1.0)
    )

    outgoing_pressure = second_outgoing / (work + second_outgoing + eps)
    incoming_pressure = second_incoming / (work + second_incoming + eps)

    bottleneck = np.sqrt(
        np.clip(outgoing_pressure, 0.0, 1.0)
        * np.clip(incoming_pressure, 0.0, 1.0)
    )

    score = (
        0.55 * locality
        + 0.30 * regret
        + 0.15 * bottleneck
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)

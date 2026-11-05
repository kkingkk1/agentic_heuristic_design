import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize = np.asarray(prize, dtype=float)
    weight = np.asarray(weight, dtype=float)

    if prize.ndim != 1:
        raise ValueError("prize must have shape (n,)")
    if weight.ndim != 2 or weight.shape[0] != prize.shape[0]:
        raise ValueError("weight must have shape (n, m)")
    if np.any(~np.isfinite(prize)) or np.any(~np.isfinite(weight)):
        raise ValueError("prize and weight must contain only finite values")
    if np.any(weight < 0):
        raise ValueError("weight must be non-negative")

    n, m = weight.shape
    if n == 0:
        return np.empty(0, dtype=float)
    if m == 0:
        return np.maximum(prize, 0.0)

    eps = np.finfo(float).eps

    total_weight = np.sum(weight, axis=0)
    scarcity = 1.0 / np.maximum(total_weight, eps)
    scarcity /= np.mean(scarcity)

    weighted_usage = np.sum(weight * scarcity[None, :], axis=1)
    max_usage = np.max(weight, axis=1)
    mean_usage = np.mean(weight, axis=1)

    usage_score = 1.0 / (eps + 0.65 * weighted_usage + 0.25 * max_usage + 0.10 * mean_usage)

    positive_prize = np.maximum(prize, 0.0)
    prize_score = np.log1p(positive_prize)
    prize_scale = np.mean(prize_score) + eps
    prize_score = prize_score / prize_scale

    return prize_score * usage_score

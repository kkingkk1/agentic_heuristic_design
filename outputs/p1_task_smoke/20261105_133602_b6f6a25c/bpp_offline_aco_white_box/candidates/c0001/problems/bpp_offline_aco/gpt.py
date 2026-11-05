import numpy as np

def heuristics_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
    demand = np.asarray(demand, dtype=float)

    if demand.ndim != 1:
        raise ValueError("demand must be a one-dimensional array")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not np.all(np.isfinite(demand)):
        raise ValueError("demand must contain only finite values")
    if np.any(demand < 0):
        raise ValueError("demand must be non-negative")

    n = demand.size
    if n == 0:
        return np.empty((0, 0), dtype=float)

    cap = float(capacity)
    sizes_i = demand[:, None]
    sizes_j = demand[None, :]
    pair_sum = sizes_i + sizes_j
    slack = cap - pair_sum

    feasible = slack >= 0.0

    # High values indicate that an item is unusually large or small
    # relative to the current instance, making it more important to match.
    mean_size = float(np.mean(demand))
    scale = float(np.std(demand)) + 1e-12
    rarity = 1.0 + np.abs(demand - mean_size) / scale
    rarity_pair = np.sqrt(rarity[:, None] * rarity[None, :])

    # Favor tight but feasible pairings.
    tightness = np.exp(-np.maximum(slack, 0.0) / (0.15 * cap + 1e-12))

    # Favor pairs with comparable sizes to avoid leaving highly fragmented
    # residual capacities after placing one item.
    balance = 1.0 - np.abs(sizes_i - sizes_j) / (cap + 1e-12)
    balance = np.clip(balance, 0.0, 1.0)

    score = tightness * (0.65 + 0.35 * balance) * rarity_pair
    score = np.where(feasible, score, -1.0)

    # An item can always be considered as a singleton if it fits.
    diagonal_value = 1.5 + rarity
    diagonal = np.arange(n)
    score[diagonal, diagonal] = np.where(
        demand <= cap,
        diagonal_value,
        -1.0
    )

    return score.astype(float, copy=False)

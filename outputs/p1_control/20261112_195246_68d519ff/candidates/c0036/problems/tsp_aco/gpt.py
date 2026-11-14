import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)

    valid = ~np.eye(n, dtype=bool)
    positive = distances[valid & (distances > 0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(positive))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    normalized = distances / scale
    kernel = np.exp(-np.square(normalized))
    kernel[~valid] = 0.0

    row_sum = kernel.sum(axis=1, keepdims=True)
    transition = np.divide(
        kernel,
        row_sum,
        out=np.zeros_like(kernel),
        where=row_sum > 0,
    )

    diffusion = transition @ transition @ transition
    diffusion = 0.5 * (diffusion + diffusion.T)
    diffusion[~valid] = 0.0

    diffusion_max = diffusion.max()
    if diffusion_max > 0:
        diffusion /= diffusion_max

    direct_max = kernel.max()
    if direct_max > 0:
        kernel /= direct_max

    score = kernel * (0.35 + 0.65 * diffusion)
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0

    maximum = score.max()
    if maximum > 0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)

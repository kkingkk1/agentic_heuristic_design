import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be a square 2D array")

    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distance_matrix)
    valid = finite & finite.T & off_diagonal

    positive = valid & (distance_matrix > 0)
    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    fallback = max(fallback, np.finfo(float).eps)

    distances = np.where(np.isfinite(distance_matrix), distance_matrix, fallback)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    eps = np.finfo(float).eps

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(distances[valid], eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0

    preference = inverse / row_max
    preference[~valid] = 0.0

    k = min(n - 1, max(2, int(np.ceil(np.log2(n)))))

    masked_preference = np.where(valid, preference, -np.inf)
    threshold = np.partition(
        masked_preference,
        kth=n - k,
        axis=1,
    )[:, n - k][:, None]

    candidate = valid & (preference >= threshold)
    candidate = candidate | candidate.T
    candidate &= valid

    nearest = np.argmax(masked_preference, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    rows = np.arange(n)
    mutual = nearest[nearest] == rows
    mutual_nearest[rows[mutual], nearest[mutual]] = 1.0
    mutual_nearest = np.maximum(mutual_nearest, mutual_nearest.T)
    mutual_nearest[~valid] = 0.0

    masked_distances = np.where(valid, distances, np.inf)
    nearest_two = np.partition(masked_distances, kth=1, axis=1)[:, :2]

    row_best = nearest_two[:, 0]
    row_second = nearest_two[:, 1]
    row_second = np.where(np.isfinite(row_second), row_second, row_best)
    row_best = np.where(np.isfinite(row_best), row_best, fallback)
    row_second = np.where(np.isfinite(row_second), row_second, fallback)

    savings = row_second[:, None] + row_second[None, :] - 2.0 * distances
    savings = np.maximum(savings, 0.0)
    savings[~valid] = 0.0

    positive_savings = savings[savings > 0]
    if positive_savings.size:
        savings_scale = max(
            float(np.percentile(positive_savings, 90)),
            eps,
        )
        savings_score = np.minimum(savings / savings_scale, 1.0)
    else:
        savings_score = np.zeros_like(savings)

    score = (
        0.60 * preference
        + 0.25 * candidate.astype(float)
        + 0.10 * mutual_nearest
        + 0.05 * savings_score
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    safe = distances.copy()
    positive = distances[valid]
    fallback = float(np.max(positive)) * 2.0 + np.finfo(float).eps
    safe[~valid] = fallback

    ranks = np.ones((n, n), dtype=float)
    for i in range(n):
        row_valid = valid[i]
        count = int(np.sum(row_valid))
        if count == 0:
            continue

        values = safe[i, row_valid]
        order = np.argsort(values, kind="mergesort")
        row_ranks = np.empty(count, dtype=float)
        row_ranks[order] = (np.arange(count, dtype=float) + 1.0) / count
        ranks[i, row_valid] = row_ranks

    local_competition = np.zeros((n, n), dtype=float)
    local_competition[valid] = 1.0 - ranks[valid]

    profile_similarity = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            common = valid[i] & valid[j]
            common[i] = False
            common[j] = False

            if np.any(common):
                a = ranks[i, common]
                b = ranks[j, common]
                difference = np.mean(np.abs(a - b))
                similarity = 1.0 - difference
            else:
                similarity = 0.0

            profile_similarity[i, j] = similarity
            profile_similarity[j, i] = similarity

    row_mean = np.zeros(n, dtype=float)
    for i in range(n):
        if np.any(valid[i]):
            row_mean[i] = np.mean(safe[i, valid[i]])

    isolation = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not (valid[i, j] or valid[j, i]):
                continue

            reference = max(row_mean[i], row_mean[j], np.finfo(float).eps)
            excess = max(safe[i, j], safe[j, i]) / reference
            penalty = 1.0 / (1.0 + excess * excess)

            isolation[i, j] = penalty
            isolation[j, i] = penalty

    rank_kernel = np.zeros((n, n), dtype=float)
    rank_kernel[valid] = np.exp(-3.0 * ranks[valid] ** 2)

    score = (
        0.30 * local_competition
        + 0.30 * profile_similarity
        + 0.20 * isolation
        + 0.20 * rank_kernel
    )

    score = 0.5 * (score + score.T)
    score[~(valid | valid.T)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score

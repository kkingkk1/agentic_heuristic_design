import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(distance_matrix)
    nonnegative = distance_matrix >= 0.0
    valid = finite & nonnegative
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    distances = np.where(valid, distance_matrix, 0.0)
    positive_valid = valid & (distances > 0.0)

    if np.any(positive_valid):
        positive_values = distances[positive_valid]
        scale = float(np.median(positive_values))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.max(positive_values))
    else:
        scale = 1.0

    eps = np.finfo(float).eps
    zero_valid = valid & (distances == 0.0)

    # 有界的距离亲和度：零距离边取最高值，避免倒数爆炸。
    distance_affinity = np.zeros((n, n), dtype=float)
    distance_affinity[valid] = scale / (scale + distances[valid] + eps)
    distance_affinity[zero_valid] = 1.0

    # 行内最近邻排名，排名越靠前分数越高。
    rank_affinity = np.zeros((n, n), dtype=float)
    nearest_rank = np.empty(n, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        nearest_rank[i] = candidates.size
        if candidates.size == 0:
            continue

        order = candidates[np.argsort(distances[i, candidates], kind="stable")]
        count = order.size
        rank_values = np.linspace(1.0, 0.0, count, endpoint=False)
        rank_affinity[i, order] = rank_values

    # 互惠偏好：一条边同时接近两端节点各自的局部最近邻时得分更高。
    mutual_affinity = np.sqrt(
        np.maximum(distance_affinity, 0.0)
        * np.maximum(distance_affinity.T, 0.0)
    )
    mutual_affinity[~valid] = 0.0

    # 每个节点的两个最近合法邻居，用于计算两邻居 savings。
    savings = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size < 2:
            continue

        order = candidates[np.argsort(distances[i, candidates], kind="stable")]
        first = float(distances[i, order[0]])
        second = float(distances[i, order[1]])

        # savings = (second_nearest - edge) / (second_nearest + edge)
        denominator = second + distances[i, order] + eps
        values = (second - distances[i, order]) / denominator
        savings[i, order] = np.maximum(values, 0.0)

    def masked_row_normalize(matrix: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(matrix, dtype=float)
        for i in range(n):
            row_mask = valid[i]
            if not np.any(row_mask):
                continue

            row = np.maximum(matrix[i, row_mask], 0.0)
            maximum = float(np.max(row))

            if maximum > eps:
                normalized[i, row_mask] = row / maximum
            else:
                normalized[i, row_mask] = 0.0

        return normalized

    score = (
        0.40 * masked_row_normalize(distance_affinity)
        + 0.25 * masked_row_normalize(rank_affinity)
        + 0.25 * masked_row_normalize(mutual_affinity)
        + 0.10 * masked_row_normalize(savings)
    )

    # 对称 TSP 使用双向平均，减少输入微小非对称造成的噪声。
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > eps:
        score /= maximum

    return score

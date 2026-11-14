import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    original_finite = np.isfinite(distance_matrix)
    valid = original_finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(distance_matrix, dtype=float)

    positive = valid & (distance_matrix > 0)
    fallback = (
        np.median(distance_matrix[positive])
        if np.any(positive)
        else 1.0
    )
    fallback = max(float(fallback), np.finfo(float).eps)

    distances = np.where(valid, distance_matrix, fallback)
    distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(distances, np.inf)

    # 基础短边偏好，并按每个节点的最佳可行边归一化。
    inverse_distance = 1.0 / distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    best_inverse = np.max(inverse_distance, axis=1, keepdims=True)
    best_inverse[best_inverse <= 0] = 1.0
    local_preference = inverse_distance / best_inverse

    # 每个节点的最近邻距离，用于构造无固定起点的节省值。
    nearest_distance = np.min(distances, axis=1)
    nearest_distance[~np.isfinite(nearest_distance)] = fallback

    savings = (
        nearest_distance[:, None]
        + nearest_distance[None, :]
        - distances
    )
    savings = np.maximum(savings, 0.0)

    savings_scale = (
        nearest_distance[:, None]
        + nearest_distance[None, :]
        + np.finfo(float).eps
    )
    savings = savings / savings_scale

    # 奖励互为近邻，同时抑制只对单个节点有利的边。
    nearest_indices = np.argmin(distances, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    rows = np.arange(n)
    mutual = nearest_indices[nearest_indices] == rows
    mutual_nearest[rows[mutual], nearest_indices[rows[mutual]]] = 1.0

    # 对非互为近邻边保留平滑兼容性，避免评分过于离散。
    geometric_scale = np.sqrt(
        nearest_distance[:, None] * nearest_distance[None, :]
    )
    geometric_scale = np.maximum(geometric_scale, np.finfo(float).eps)
    compatibility = np.exp(-distances / geometric_scale)
    compatibility[~np.isfinite(compatibility)] = 0.0

    compatibility = 0.7 * compatibility + 0.3 * mutual_nearest

    # 结构信号为主，基础短边信号保留探索作用。
    structured = (
        0.50 * local_preference
        + 0.30 * savings
        + 0.20 * compatibility
    )
    score = 0.88 * structured + 0.12 * local_preference

    score[~original_finite] = 0.0
    np.fill_diagonal(score, 0.0)

    # TSP 距离矩阵通常为对称矩阵；对输出做对称化以保持边的一致先验。
    score = 0.5 * (score + score.T)

    maximum = np.max(score)
    if maximum > 0:
        score = score / maximum

    return score.reshape(distance_matrix.shape)

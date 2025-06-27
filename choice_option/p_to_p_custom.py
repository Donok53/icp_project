import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


def compute_transformation_svd(source, target):
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)

    source_centered = source - source_center
    target_centered = target - target_center

    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection 제거 (det(R) = -1인 경우)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = target_center - R @ source_center
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def run_p2p_icp(source_pcd, target_pcd, init_trans=np.eye(4), max_iter=20, tol=1e-6):
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)

    T_total = init_trans.copy()
    src_transformed = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]

    for i in range(max_iter):
        tree = KDTree(tgt_pts)
        dists, idxs = tree.query(src_transformed)
        target_corr = tgt_pts[idxs]

        T_delta = compute_transformation_svd(src_transformed, target_corr)
        src_transformed = (T_delta[:3, :3] @ src_transformed.T).T + T_delta[:3, 3]
        T_total = T_delta @ T_total

        if np.linalg.norm(T_delta[:3, 3]) < tol:
            break

    # 피트니스 및 RMSE 계산
    final_dists = np.linalg.norm(src_transformed - target_corr, axis=1)
    inlier_mask = final_dists < 2.0
    fitness = np.sum(inlier_mask) / len(final_dists)
    rmse = (
        np.sqrt(np.mean(final_dists[inlier_mask] ** 2))
        if np.any(inlier_mask)
        else float("inf")
    )

    return T_total, fitness, rmse

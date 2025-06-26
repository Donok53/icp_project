import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def compute_line_directions(pcd, k=10):
    pts = np.asarray(pcd.points)
    tree = KDTree(pts)
    directions = []

    for i in range(len(pts)):
        _, idxs = tree.query(pts[i], k=k)
        neighbors = pts[idxs]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, -1]  # 주성분 (가장 큰 고유값)
        directions.append(v)

    return np.array(directions)

def run_point_to_line_icp_custom(source_pcd, target_pcd, init_trans=np.eye(4), max_iter=20, tol=1e-6):
    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)

    # 라인 방향 벡터 필요
    line_directions = compute_line_directions(target_pcd)

    T_total = init_trans.copy()
    source_transformed = (T_total[:3, :3] @ source_pts.T).T + T_total[:3, 3]
    tree = KDTree(target_pts)

    for i in range(max_iter):
        dists, idxs = tree.query(source_transformed)
        corr_tgt = target_pts[idxs]
        corr_dirs = line_directions[idxs]

        H = np.zeros((6, 6))
        g = np.zeros((6, 1))

        for p, q, v in zip(source_transformed, corr_tgt, corr_dirs):
            v = v / np.linalg.norm(v)
            r = np.cross((p - q), v).reshape(3, 1)  # 3x1 잔차

            J = np.zeros((3, 6))
            J[:, :3] = -skew(np.cross(v, p))      # 회전에 대한 도함수
            J[:, 3:] = -skew(v)                   # 병진에 대한 도함수

            H += J.T @ J
            g += J.T @ r

        try:
            dx, *_ = np.linalg.lstsq(H, -g, rcond=None)
        except np.linalg.LinAlgError:
            print("[WARN] Singular matrix during optimization.")
            break

        delta = dx.flatten()
        if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
            print("[WARN] Invalid delta (NaN/Inf)")
            break

        R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
        t_delta = delta[3:]
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta

        source_transformed = (R_delta @ source_transformed.T).T + t_delta
        T_total = T_delta @ T_total

        if np.linalg.norm(delta) < tol:
            break

    # 최종 정합 품질 평가
    final_corr_tgt = target_pts[idxs]
    final_corr_dirs = line_directions[idxs]
    dist_vecs = np.cross((source_transformed - final_corr_tgt), final_corr_dirs)
    dists = np.linalg.norm(dist_vecs, axis=1)

    inliers = dists < 1.0
    fitness = np.sum(inliers) / len(dists)
    rmse = np.sqrt(np.mean(dists[inliers] ** 2)) if np.any(inliers) else float('inf')

    return T_total, fitness, rmse

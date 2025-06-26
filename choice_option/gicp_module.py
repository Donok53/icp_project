import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

def skew(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def compute_covariances(pcd, max_nn=30):
    pts = np.asarray(pcd.points)
    tree = KDTree(pts)
    covariances = []
    for pt in pts:
        dists, idxs = tree.query(pt, k=max_nn)
        if len(idxs.shape) == 0 or len(idxs) < 3:
            cov = np.eye(3) * 1e-2
        else:
            neighbors = pts[idxs] - pt
            cov = np.cov(neighbors.T) + np.eye(3) * 1e-2
        covariances.append(cov)
    return np.stack(covariances)

def run_gicp(source_pcd, target_pcd, init_trans, optimizer='least_squares', max_iter=20, tol=1e-6):
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)
    src_covs = compute_covariances(source_pcd)
    tgt_covs = compute_covariances(target_pcd)

    T_total = init_trans.copy()
    src_transformed = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]

    for i in range(max_iter):
        tree = KDTree(tgt_pts)
        dists, idxs = tree.query(src_transformed)
        tgt_corr = tgt_pts[idxs]
        tgt_corr_cov = tgt_covs[idxs]

        H, g = np.zeros((6, 6)), np.zeros((6, 1))
        for p, q, cov_p, cov_q in zip(src_transformed, tgt_corr, src_covs, tgt_corr_cov):
            R = T_total[:3, :3]
            C = cov_p + R @ cov_q @ R.T + np.eye(3) * 1e-2
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                continue

            r = (q - p).reshape(3, 1)
            J = np.zeros((3, 6))
            J[:, :3] = -skew(p)
            J[:, 3:] = -np.eye(3)

            H += J.T @ C_inv @ J
            g += J.T @ C_inv @ r

        try:
            dx, *_ = np.linalg.lstsq(H, -g, rcond=None)
        except np.linalg.LinAlgError:
            print("[WARN] Singular matrix")
            return None, 0.0, float('inf')

        delta = dx.flatten()
        if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
            print("[WARN] Invalid delta")
            return None, 0.0, float('inf')

        R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
        t_delta = delta[3:]
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta

        src_transformed = (R_delta @ src_transformed.T).T + t_delta
        T_total = T_delta @ T_total

        if np.linalg.norm(delta) < tol:
            break

    # 피트니스 및 RMSE 계산
    final_dists = np.linalg.norm(src_transformed - tgt_corr, axis=1)
    inlier_mask = final_dists < 2.0  # threshold 예시
    fitness = np.sum(inlier_mask) / len(final_dists)
    rmse = np.sqrt(np.mean(final_dists[inlier_mask] ** 2)) if np.any(inlier_mask) else float('inf')

    return T_total, fitness, rmse

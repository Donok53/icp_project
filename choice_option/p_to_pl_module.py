# icp_p2pl_module.py
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

def skew(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def compute_normals(points, max_nn=30):
    tree = KDTree(points)
    normals = []
    for i in range(len(points)):
        dists, idxs = tree.query(points[i], k=max_nn)
        neighbors = points[idxs] - points[i]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals.append(eigvecs[:, 0])
    return np.stack(normals)

def run_p2pl_icp(source_pcd, target_pcd, init_trans=np.eye(4), optimizer='least_squares', max_iter=20, tol=1e-6):
    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)
    target_normals = compute_normals(target_pts)

    T = init_trans.copy()
    src = (T[:3, :3] @ source_pts.T).T + T[:3, 3]

    for _ in range(max_iter):
        tree = KDTree(target_pts)
        dists, idxs = tree.query(src)
        corr_target = target_pts[idxs]
        corr_normals = target_normals[idxs]

        H = np.zeros((6, 6))
        b = np.zeros((6, 1))

        for p, q, n in zip(src, corr_target, corr_normals):
            r = (p - q).dot(n)
            J = np.zeros((1, 6))
            J[0, :3] = (p @ skew(n)).T
            J[0, 3:] = n

            H += J.T @ J
            b += J.T * r

        # ------------------- 여기서 optimizer 적용 -------------------
        try:
            if optimizer == 'lm':
                lambda_ = 1e-3
                delta = -np.linalg.solve(H + lambda_ * np.eye(6), b)
            elif optimizer in ['least_squares', 'gauss_newton']:
                delta = -np.linalg.solve(H, b)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        except np.linalg.LinAlgError:
            print("[WARN] Singular matrix")
            break

        angle = delta[:3].flatten()
        trans = delta[3:].flatten()
        R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(angle)
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = trans

        src = (R_delta @ src.T).T + trans
        T = T_delta @ T

        if np.linalg.norm(delta) < tol:
            break

    # 정합 품질 계산
    final_dists = np.linalg.norm(src - corr_target, axis=1)
    inliers = final_dists < 2.0
    fitness = np.sum(inliers) / len(final_dists)
    rmse = np.sqrt(np.mean(final_dists[inliers] ** 2)) if np.any(inliers) else float('inf')

    return T, fitness, rmse


import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def run_point_to_line_icp_custom(source_pcd, target_pcd, init_trans=np.eye(4), optimizer='least_squares', max_iter=20, tol=1e-6):
    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)

    # Normals 필요
    target_normals = np.asarray(target_pcd.normals)
    if len(target_normals) == 0:
        raise RuntimeError("Target point cloud must have normals. Run estimate_normals first.")

    T_total = init_trans.copy()
    source_transformed = (T_total[:3, :3] @ source_pts.T).T + T_total[:3, 3]

    tree = KDTree(target_pts)

    for i in range(max_iter):
        dists, idxs = tree.query(source_transformed)
        corr_tgt = target_pts[idxs]
        corr_normals = target_normals[idxs]

        H = np.zeros((6, 6))
        g = np.zeros((6, 1))

        for p, q, n in zip(source_transformed, corr_tgt, corr_normals):
            r = n @ (q - p)
            J = np.zeros((1, 6))
            J[0, :3] = n @ skew(p)
            J[0, 3:] = -n

            H += J.T @ J
            g += J.T * r

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
    dists = np.abs(np.sum((corr_tgt - source_transformed) * corr_normals, axis=1))
    inliers = dists < 1.0
    fitness = np.sum(inliers) / len(dists)
    rmse = np.sqrt(np.mean(dists[inliers] ** 2)) if np.any(inliers) else float('inf')

    return T_total, fitness, rmse

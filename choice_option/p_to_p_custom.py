import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

# ----------------------------------------
# Utility functions
# ----------------------------------------
def load_point_cloud(file_path, voxel_size=0.2):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size)


def compute_transformation_svd(source, target):
    # Compute closed-form SVD-based rigid transform
    src_center = np.mean(source, axis=0)
    tgt_center = np.mean(target, axis=0)
    src_centered = source - src_center
    tgt_centered = target - tgt_center
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_center - R @ src_center
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def skew(v):
    # Skew-symmetric matrix for cross-product
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# ----------------------------------------
# ICP: Point-to-Point with optimizer options
# ----------------------------------------
def run_p2p_icp(source_pcd, target_pcd,
                init_trans=np.eye(4),
                max_iter=20,
                tol=1e-6,
                optimizer='svd',  # 'svd', 'least_squares', 'gauss_newton', 'lm'
                lambda_=1e-3):
    # Extract raw points
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)

    # Initialize total transform and best-tracking
    T_total = init_trans.copy()
    best_T = T_total.copy()
    best_rmse = float('inf')

    # Initial transformed source
    src_trans = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]

    if optimizer == 'svd':
        for _ in range(max_iter):
            tree = KDTree(tgt_pts)
            dists, idxs = tree.query(src_trans)
            tgt_corr = tgt_pts[idxs]

            # SVD update
            T_delta = compute_transformation_svd(src_trans, tgt_corr)
            R_delta = T_delta[:3, :3]
            t_delta = T_delta[:3, 3]
            src_trans = (R_delta @ src_trans.T).T + t_delta
            T_total = T_delta @ T_total

            # RMSE and best update
            d = np.linalg.norm(src_trans - tgt_corr, axis=1)
            mask = d < 2.0
            rmse_i = np.sqrt(np.mean(d[mask]**2)) if np.any(mask) else float('inf')
            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_T = T_total.copy()

            if np.linalg.norm(t_delta) < tol:
                break
    else:
        # iterative optimizer
        x = np.zeros(6)
        for _ in range(max_iter):
            # extract R, t
            omega = x[:3]
            theta = np.linalg.norm(omega)
            if theta < 1e-12:
                R_curr = np.eye(3)
            else:
                k = omega / theta
                K = skew(k)
                R_curr = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
            t_curr = x[3:]

            # transform
            src_trans = (R_curr @ src_pts.T).T + t_curr
            tree = KDTree(tgt_pts)
            dists, idxs = tree.query(src_trans)
            mask_corr = dists < np.inf
            P_corr = src_trans[mask_corr]
            Q_corr = tgt_pts[idxs[mask_corr]]

            # Hessian & gradient
            H = np.zeros((6,6)); g = np.zeros(6)
            for p_i, q_i in zip(P_corr, Q_corr):
                r = p_i - q_i
                J_i = np.zeros((3,6))
                J_i[:, :3] = -skew(p_i)
                J_i[:, 3:] = -np.eye(3)
                H += J_i.T @ J_i
                g += J_i.T @ r

            # solve dx
            if optimizer == 'lm':
                H_lm = H + lambda_ * np.eye(6)
                dx = -np.linalg.solve(H_lm, g)
            else:
                dx = -np.linalg.solve(H, g)

            # update state
            if np.linalg.norm(dx) < tol:
                break
            x += dx

            # accumulate transform
            omega = dx[:3]; theta = np.linalg.norm(omega)
            if theta < 1e-12:
                R_d = np.eye(3)
            else:
                k = omega / theta; K = skew(k)
                R_d = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
            t_d = dx[3:]
            T_delta = np.eye(4); T_delta[:3,:3] = R_d; T_delta[:3,3] = t_d
            T_total = T_delta @ T_total

            # rmse_i compute
            src_iter = (T_total[:3,:3] @ src_pts.T).T + T_total[:3,3]
            d2, idx2 = KDTree(tgt_pts).query(src_iter)
            mask2 = d2 < 2.0
            rmse_i = np.sqrt(np.mean(d2[mask2]**2)) if np.any(mask2) else float('inf')
            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_T = T_total.copy()

    # --- rollback to best iteration ---
    T_total = best_T.copy()

    # final src recompute
    final_src = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]
    # fresh correspondence and evaluation
    treef = KDTree(tgt_pts)
    d_final, _ = treef.query(final_src)
    mask_f = d_final < 2.0
    fitness = np.sum(mask_f) / len(d_final)
    rmse = np.sqrt(np.mean(d_final[mask_f]**2)) if np.any(mask_f) else float('inf')

    return T_total, fitness, rmse


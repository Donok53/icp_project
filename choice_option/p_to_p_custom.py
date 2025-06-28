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

    # Initialize total transform
    T_total = init_trans.copy()
    src_transformed = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]

    if optimizer == 'svd':
        # Closed-form SVD loop
        for i in range(max_iter):
            tree = KDTree(tgt_pts)
            dists, idxs = tree.query(src_transformed)
            tgt_corr = tgt_pts[idxs]
            T_delta = compute_transformation_svd(src_transformed, tgt_corr)
            src_transformed = (T_delta[:3, :3] @ src_transformed.T).T + T_delta[:3, 3]
            T_total = T_delta @ T_total
            if np.linalg.norm(T_delta[:3, 3]) < tol:
                break
    else:
        # Iterative optimization (GN or LM)
        # State x = [omega_x, omega_y, omega_z, t_x, t_y, t_z]
        x = np.zeros(6)
        for i in range(max_iter):
            # Compute rotation R and translation t from x
            omega = x[:3]
            theta = np.linalg.norm(omega)
            if theta < 1e-12:
                R = np.eye(3)
            else:
                k = omega / theta
                K = skew(k)
                R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            t = x[3:]

            # Transform source
            P_trans = (R @ src_pts.T).T + t
            tree = KDTree(tgt_pts)
            dists, idxs = tree.query(P_trans)
            mask = dists < np.inf
            P_corr = P_trans[mask]
            Q_corr = tgt_pts[idxs[mask]]

            # Build Hessian H and gradient g
            H = np.zeros((6, 6))
            g = np.zeros(6)
            for p_i, q_i in zip(P_corr, Q_corr):
                r = p_i - q_i
                J_i = np.zeros((3, 6))
                J_i[:, :3] = -skew(p_i)
                J_i[:, 3:] = -np.eye(3)
                H += J_i.T @ J_i
                g += J_i.T @ r

            # Solve for dx
            if optimizer == 'lm':
                H_lm = H + lambda_ * np.eye(6)
                dx = -np.linalg.solve(H_lm, g)
            else:
                dx = -np.linalg.solve(H, g)

            x += dx
            if np.linalg.norm(dx) < tol:
                break

        # After convergence, form final delta transform
        omega = x[:3]
        theta = np.linalg.norm(omega)
        if theta < 1e-12:
            R = np.eye(3)
        else:
            k = omega / theta
            K = skew(k)
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        t = x[3:]
        T_delta = np.eye(4)
        T_delta[:3, :3] = R
        T_delta[:3, 3] = t
        src_transformed = (T_delta[:3, :3] @ src_transformed.T).T + T_delta[:3, 3]
        T_total = T_delta @ T_total

    # Compute fitness and RMSE
    tree_final = KDTree(tgt_pts)
    final_dists, _ = tree_final.query(src_transformed)
    inlier_mask = final_dists < 2.0
    fitness = np.sum(inlier_mask) / len(final_dists)
    rmse = np.sqrt(np.mean(final_dists[inlier_mask] ** 2)) if np.any(inlier_mask) else float('inf')

    return T_total, fitness, rmse

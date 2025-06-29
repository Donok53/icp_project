import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from choice_option.p_to_p_custom import compute_transformation_svd


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_covariances(pcd, max_nn=30):
    pts = np.asarray(pcd.points)
    tree = KDTree(pts)
    covariances = []
    for pt in pts:
        dists, idxs = tree.query(pt, k=max_nn)
        if idxs is None or len(idxs) < 3:
            cov = np.eye(3) * 1e-6
        else:
            neighbors = pts[idxs] - pt
            cov = np.cov(neighbors.T) + np.eye(3) * 1e-6
        covariances.append(cov)
    return np.stack(covariances)


def run_gicp(
    source_pcd,
    target_pcd,
    init_trans=np.eye(4),
    optimizer="least_squares",
    max_iter=20,
    tol=1e-6,
):
    # Prepare data
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)
    src_covs = compute_covariances(source_pcd)
    tgt_covs = compute_covariances(target_pcd)

    # Initial transform
    T_total = init_trans.copy()
    src = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]

    for _ in range(max_iter):
        # Find nearest neighbors
        tree = KDTree(tgt_pts)
        dists, idxs = tree.query(src)
        corr_src = src
        corr_tgt = tgt_pts[idxs]
        corr_src_cov = src_covs
        corr_tgt_cov = tgt_covs[idxs]

        # SVD closed-form on point correspondences
        if optimizer == "svd":
            T_delta = compute_transformation_svd(corr_src, corr_tgt)
            R_delta = T_delta[:3, :3]
            t_delta = T_delta[:3, 3]

            src = (R_delta @ src.T).T + t_delta
            T_total = T_delta @ T_total

            if np.linalg.norm(t_delta) < tol:
                break
            else:
                continue

        # Build Hessian and gradient for G-ICP
        H = np.zeros((6, 6))
        g = np.zeros((6, 1))
        R_prev = T_total[:3, :3]
        for i, (p, q) in enumerate(zip(corr_src, corr_tgt)):
            C = corr_src_cov[i] + R_prev @ corr_tgt_cov[i] @ R_prev.T + np.eye(3) * 1e-6
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

        # Solve for update
        try:
            if optimizer == "lm":
                lambda_ = 1e-4
                dx = -np.linalg.solve(H + lambda_ * np.eye(6), g)
            elif optimizer in ["least_squares", "gauss_newton"]:
                dx = -np.linalg.solve(H, g)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        except np.linalg.LinAlgError:
            print("[WARN] Singular matrix")
            break

        delta = dx.flatten()
        R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
        t_delta = delta[3:]
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta

        # Apply update
        src = (R_delta @ src.T).T + t_delta
        T_total = T_delta @ T_total

        if np.linalg.norm(delta) < tol:
            break

    # Evaluate fitness and rmse
    final_dists = np.linalg.norm(src - corr_tgt, axis=1)
    inliers = final_dists < 1.0
    fitness = np.sum(inliers) / len(final_dists)
    rmse = (
        np.sqrt(np.mean(final_dists[inliers] ** 2)) if np.any(inliers) else float("inf")
    )

    return T_total, fitness, rmse

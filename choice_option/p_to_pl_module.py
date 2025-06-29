import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from choice_option.p_to_p_custom import compute_transformation_svd


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


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


def run_p2pl_icp(
    source_pcd, target_pcd, init_trans=np.eye(4), optimizer="svd", max_iter=20, tol=1e-6
):
    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)
    target_normals = compute_normals(target_pts)

    T_total = init_trans.copy()
    src = (T_total[:3, :3] @ source_pts.T).T + T_total[:3, 3]

    for _ in range(max_iter):
        tree = KDTree(target_pts)
        dists, idxs = tree.query(src)
        corr_src = src
        corr_target = target_pts[idxs]
        corr_normals = target_normals[idxs]

        # SVD closed-form branch
        if optimizer == "svd":
            diffs = corr_src - corr_target  # (N×3)
            dists_n = np.sum(diffs * corr_normals, axis=1, keepdims=True)  # (N×1)
            proj_q = corr_target + corr_normals * dists_n  # (N×3)

            T_delta = compute_transformation_svd(corr_src, proj_q)
            R_delta = T_delta[:3, :3]
            t_delta = T_delta[:3, 3]

            src = (R_delta @ src.T).T + t_delta
            T_total = T_delta @ T_total

            if np.linalg.norm(t_delta) < tol:
                break
            else:
                continue

        # LS/GN/LM branch
        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        for p, q, n in zip(corr_src, corr_target, corr_normals):
            r = (p - q).dot(n)
            J = np.zeros((1, 6))
            J[0, :3] = (p @ skew(n)).T
            J[0, 3:] = n

            H += J.T @ J
            b += J.T * r

        try:
            if optimizer == "lm":
                lambda_ = 1e-3
                delta = -np.linalg.solve(H + lambda_ * np.eye(6), b)
            elif optimizer in ["least_squares", "gauss_newton"]:
                delta = -np.linalg.solve(H, b)
            else:
                raise ValueError("Unsupported optimizer: {}".format(optimizer))
        except np.linalg.LinAlgError:
            print("[WARN] Singular matrix during optimization.")
            break

        angle = delta[:3].flatten()
        trans = delta[3:].flatten()
        R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(angle)
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = trans

        src = (R_delta @ src.T).T + trans
        T_total = T_delta @ T_total

        if np.linalg.norm(delta) < tol:
            break

    # 평가
    final_corr = np.cross((src - corr_target), corr_normals)
    dists = np.linalg.norm(final_corr, axis=1)
    inliers = dists < 2.0
    fitness = np.sum(inliers) / len(dists)
    rmse = np.sqrt(np.mean(dists[inliers] ** 2)) if np.any(inliers) else float("inf")

    return T_total, fitness, rmse

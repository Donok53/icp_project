import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from choice_option.p_to_p_custom import compute_transformation_svd  # ➊


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def compute_line_directions(pcd, k=10):
    pts = np.asarray(pcd.points)
    tree = KDTree(pts)
    directions = []
    for i in range(len(pts)):
        _, idxs = tree.query(pts[i], k=k)
        neighbors = pts[idxs]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        directions.append(eigvecs[:, -1])
    return np.array(directions)


def run_point_to_line_icp_custom(
    source_pcd, target_pcd, init_trans=np.eye(4),
    optimizer="svd", max_iter=20, tol=1e-6
):
    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)
    line_dirs = compute_line_directions(target_pcd)

    T_total = init_trans.copy()
    best_T = T_total.copy()
    best_rmse = float('inf')

    src = (T_total[:3, :3] @ source_pts.T).T + T_total[:3, 3]
    tree = KDTree(target_pts)

    for _ in range(max_iter):
        # ➋ correspondence
        dists, idxs = tree.query(src)
        corr_src = src              # (N,3)
        corr_q = target_pts[idxs]   # (N,3)
        corr_v = line_dirs[idxs]    # (N,3)

        # —————— SVD closed-form 분기 ——————
        if optimizer == "svd":
            # 1) src를 line에 orthogonal projection
            diffs = corr_src - corr_q                # (N,3)
            dots = np.sum(diffs * corr_v, axis=1, keepdims=True)  # (N,1)
            proj_q = corr_q + corr_v * dots          # (N,3)
            # closed-form SVD 계산
            T_delta = compute_transformation_svd(corr_src, proj_q)
            R_delta = T_delta[:3, :3]
            t_delta = T_delta[:3, 3]
            src = (R_delta @ src.T).T + t_delta
            T_total = T_delta @ T_total
            # RMSE 평가 및 최적값 갱신
            final_corr = np.cross((src - corr_q), corr_v)
            d = np.linalg.norm(final_corr, axis=1)
            inl = d < 2.0
            rmse_i = np.sqrt(np.mean(d[inl]**2)) if np.any(inl) else float('inf')
            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_T = T_total.copy()
            # 수렴 체크
            if np.linalg.norm(t_delta) < tol:
                break
            else:
                continue
        # ————————————————————————————————

        # 기존 Least‐Squares / Gauss‐Newton / LM 분기
        H = np.zeros((6, 6))
        g = np.zeros((6, 1))
        for p, q, v in zip(corr_src, corr_q, corr_v):
            v = v / np.linalg.norm(v)
            r = np.cross((p - q), v).reshape(3, 1)
            J = np.zeros((3, 6))
            J[:, :3] = -skew(np.cross(v, p))
            J[:, 3:] = -skew(v)
            H += J.T @ J
            g += J.T @ r

        try:
            if optimizer == "lm":
                λ = 1e-3
                dx = -np.linalg.solve(H + λ * np.eye(6), g)
            else:
                dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("[WARN] Singular matrix")
            break

        delta = dx.flatten()
        R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
        t_delta = delta[3:]
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta

        src = (R_delta @ src.T).T + t_delta
        T_total = T_delta @ T_total

        # RMSE 평가 및 최적값 갱신
        final_corr = np.cross((src - corr_q), corr_v)
        d = np.linalg.norm(final_corr, axis=1)
        inl = d < 2.0
        rmse_i = np.sqrt(np.mean(d[inl]**2)) if np.any(inl) else float('inf')
        if rmse_i < best_rmse:
            best_rmse = rmse_i
            best_T = T_total.copy()

        if np.linalg.norm(delta) < tol:
            break

    # --- 최적 이터레이션으로 롤백 ---
    T_total = best_T.copy()

    # 1) src 재계산 (best_T 기준)
    src = (T_total[:3, :3] @ source_pts.T).T + T_total[:3, 3]
    # 2) fresh correspondence
    dists, idxs = tree.query(src)
    corr_q = target_pts[idxs]
    corr_v = line_dirs[idxs]   # ← target_normals가 아니라 line_dirs 사용
    # 3) 최종 inlier/rmse 평가
    final_corr = np.cross((src - corr_q), corr_v)
    d = np.linalg.norm(final_corr, axis=1)
    inliers = d < 2.0
    fitness = np.sum(inliers) / len(d)
    rmse = np.sqrt(np.mean(d[inliers]**2)) if np.any(inliers) else float('inf')

    return T_total, fitness, rmse

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
    source_pcd,
    target_pcd,
    init_trans=np.eye(4),
    optimizer="svd",
    max_iter=20,
    tol=1e-6,
    lambda_=1e-3   # LM 초기 댐핑 파라미터
):
    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)
    line_dirs  = compute_line_directions(target_pcd)

    T_total   = init_trans.copy()
    best_T    = T_total.copy()
    best_rmse = float('inf')

    # 초기 src 위치
    src = (T_total[:3,:3] @ source_pts.T).T + T_total[:3,3]
    tree = KDTree(target_pts)

    # LM용 λ 초기화
    if optimizer == "lm":
        lam = lambda_

    for iter_idx in range(max_iter):
        # ➋ correspondence
        dists, idxs   = tree.query(src)
        corr_src      = src
        corr_q        = target_pts[idxs]
        corr_v        = line_dirs[idxs]

        # —————— SVD closed-form 분기 ——————
        if optimizer == "svd":
            diffs    = corr_src - corr_q
            dots     = np.sum(diffs * corr_v, axis=1, keepdims=True)
            proj_q   = corr_q + corr_v * dots
            T_delta  = compute_transformation_svd(corr_src, proj_q)
            R_delta  = T_delta[:3,:3]
            t_delta  = T_delta[:3,3]
            src      = (R_delta @ src.T).T + t_delta
            T_total  = T_delta @ T_total

            # RMSE 업데이트
            final_corr = np.cross((src - corr_q), corr_v)
            d = np.linalg.norm(final_corr, axis=1)
            inl = d < 2.0
            rmse_i = np.sqrt(np.mean(d[inl]**2)) if np.any(inl) else float('inf')
            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_T    = T_total.copy()

            if np.linalg.norm(t_delta) < tol:
                break
            else:
                continue
        # ————————————————————————————————

        # Hessian & gradient
        H = np.zeros((6,6))
        g = np.zeros((6,1))
        for p, q, v in zip(corr_src, corr_q, corr_v):
            v = v / np.linalg.norm(v)
            r = np.cross((p - q), v).reshape(3,1)
            J = np.zeros((3,6))
            J[:, :3] = -skew(np.cross(v, p))
            J[:, 3:] = -skew(v)
            H += J.T @ J
            g += J.T @ r

        # ——— LM 스케줄링 분기 ———
        if optimizer == "lm":
            # 비용 함수 정의
            def compute_residual(T):
                pts = (T[:3,:3] @ source_pts.T).T + T[:3,3]
                d, _ = KDTree(target_pts).query(pts)
                return np.mean(d**2)

            old_cost = compute_residual(T_total)
            while True:
                try:
                    H_lm = H + lam * np.eye(6)
                    dx   = -np.linalg.solve(H_lm, g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue

                delta = dx.flatten()
                R_cand = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
                t_cand = delta[3:]
                T_cand = np.eye(4)
                T_cand[:3,:3] = R_cand
                T_cand[:3,3]  = t_cand
                T_new  = T_cand @ T_total

                new_cost = compute_residual(T_new)
                if new_cost < old_cost:
                    # 개선 → 수용 & λ 감소
                    T_total = T_new
                    src     = (R_cand @ src.T).T + t_cand
                    lam    *= 0.1
                    delta   = dx.flatten()
                    break
                else:
                    # 악화 → λ 증가
                    lam    *= 10
                    if lam > 1e12:
                        T_total = T_new
                        src     = (R_cand @ src.T).T + t_cand
                        delta   = dx.flatten()
                        break

        else:
            # pure Gauss–Newton
            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print("[WARN] Singular matrix")
                break
            delta = dx.flatten()
            R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
            t_delta = delta[3:]
            T_delta = np.eye(4)
            T_delta[:3,:3] = R_delta
            T_delta[:3,3]  = t_delta
            src      = (R_delta @ src.T).T + t_delta
            T_total  = T_delta @ T_total

        # RMSE 업데이트
        final_corr = np.cross((src - corr_q), corr_v)
        d = np.linalg.norm(final_corr, axis=1)
        inl = d < 2.0
        rmse_i = np.sqrt(np.mean(d[inl]**2)) if np.any(inl) else float('inf')
        if rmse_i < best_rmse:
            best_rmse = rmse_i
            best_T    = T_total.copy()

        # 수렴 검사
        if np.linalg.norm(delta) < tol:
            print(f"[INFO] Converged after {iter_idx+1} iters")
            break

    # 최적 이터레이션으로 롤백
    T_total = best_T.copy()

    # 최종 평가
    src = (T_total[:3,:3] @ source_pts.T).T + T_total[:3,3]
    dists, idxs = tree.query(src)
    corr_q = target_pts[idxs]
    corr_v = line_dirs[idxs]
    final_corr = np.cross((src - corr_q), corr_v)
    d = np.linalg.norm(final_corr, axis=1)
    inliers = d < 2.0
    fitness = np.sum(inliers)/len(d)
    rmse    = np.sqrt(np.mean(d[inliers]**2)) if np.any(inliers) else float('inf')

    return T_total, fitness, rmse
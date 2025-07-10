import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from choice_option.p_to_p_custom import compute_transformation_svd


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def compute_covariances(pcd, max_nn=10):
    pts = np.asarray(pcd.points)
    tree = KDTree(pts)
    covariances = []
    insufficient_neighbors = 0
    for pt in pts:
        dists, idxs = tree.query(pt, k=max_nn)
        if idxs is None or len(idxs) < 3:
            cov = np.eye(3) * 1e-6
            insufficient_neighbors += 1
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
    max_iter=50,
    tol=1e-6,
):
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)
    src_covs = compute_covariances(source_pcd)
    tgt_covs = compute_covariances(target_pcd)

    T_total   = init_trans.copy()
    best_T    = T_total.copy()
    best_rmse = float('inf')

    # 초기 source 위치
    src = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]

    # LM 전용 댐핑 파라미터 초기화
    if optimizer == "lm":
        lambda_ = 1e-3

    for iter_idx in range(max_iter):
        # 1) 최근접 대응
        tree = KDTree(tgt_pts)
        dists, idxs = tree.query(src)
        corr_src      = src
        corr_tgt      = tgt_pts[idxs]
        corr_src_cov  = src_covs
        corr_tgt_cov  = tgt_covs[idxs]

        # 2) SVD 분기
        if optimizer == "svd":
            T_delta = compute_transformation_svd(corr_src, corr_tgt)
            R_delta = T_delta[:3, :3]; t_delta = T_delta[:3, 3]
            src     = (R_delta @ src.T).T + t_delta
            T_total = T_delta @ T_total
            if np.linalg.norm(t_delta) < tol:
                break
            else:
                continue

        # 3) Hessian H, gradient g 계산
        H = np.zeros((6, 6))
        g = np.zeros((6, 1))
        R_prev = T_total[:3, :3]
        for i, (p, q) in enumerate(zip(corr_src, corr_tgt)):
            C = corr_src_cov[i] + R_prev @ corr_tgt_cov[i] @ R_prev.T + np.eye(3)*1e-6
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                continue
            r = (q - p).reshape(3,1)
            J = np.zeros((3,6))
            J[:, :3] = -skew(p)
            J[:, 3:] = -np.eye(3)
            H += J.T @ C_inv @ J
            g += J.T @ C_inv @ r

        # 4) 업데이트 (LM vs GN)
        if optimizer == "lm":
            # 비용 함수
            def compute_residual(T):
                pts = (T[:3,:3] @ src_pts.T).T + T[:3,3]
                d, _ = KDTree(tgt_pts).query(pts)
                return np.mean(d**2)

            old_cost = compute_residual(T_total)
            while True:
                try:
                    H_lm = H + lambda_ * np.eye(6)
                    dx   = -np.linalg.solve(H_lm, g)
                except np.linalg.LinAlgError:
                    lambda_ *= 10
                    continue

                delta = dx.flatten()
                R_cand = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
                t_cand = delta[3:]
                T_cand = np.eye(4); T_cand[:3,:3]=R_cand; T_cand[:3,3]=t_cand
                T_new  = T_cand @ T_total

                new_cost = compute_residual(T_new)
                if new_cost < old_cost:
                    # 개선됨 → 수용 & λ 줄이기
                    T_total = T_new
                    src     = (R_cand @ src.T).T + t_cand
                    lambda_ *= 0.1
                    break
                else:
                    # 악화됨 → λ 키우고 재시도
                    lambda_ *= 10
                    if lambda_ > 1e12:
                        # 너무 커지면 강제 수용
                        T_total = T_new
                        src     = (R_cand @ src.T).T + t_cand
                        break

        else:
            # pure Gauss-Newton
            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print("[WARN] Hessian singular")
                break
            delta   = dx.flatten()
            R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
            t_delta = delta[3:]
            T_delta = np.eye(4); T_delta[:3,:3]=R_delta; T_delta[:3,3]=t_delta
            src     = (R_delta @ src.T).T + t_delta
            T_total = T_delta @ T_total

        # 5) 수렴 체크
        if np.linalg.norm(delta) < tol:
            print(f"[INFO] Converged after {iter_idx+1} iterations")
            break

        # 6) best pose 기록 (원래 로직)
        pts_trans = src
        d = np.linalg.norm(pts_trans - corr_tgt, axis=1)
        rmse_i = np.sqrt(np.mean(d[d<2.0]**2)) if np.any(d<2.0) else float('inf')
        if rmse_i < best_rmse:
            best_rmse = rmse_i
            best_T    = T_total.copy()

    # 최종 rollback 및 평가 (원래 로직 유지)
    T_total = best_T.copy()
    final_src = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]
    tree = KDTree(tgt_pts)
    dists, idxs = tree.query(final_src)
    final_tgt = tgt_pts[idxs]
    d = np.linalg.norm(final_src - final_tgt, axis=1)
    inliers = d < 2.0
    fitness = np.sum(inliers)/len(d)
    rmse    = np.sqrt(np.mean(d[inliers]**2)) if np.any(inliers) else float('inf')

    return T_total, fitness, rmse

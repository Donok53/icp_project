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
    print(f"[GICP] Covariance: insufficient neighbors = {insufficient_neighbors}/{len(pts)} ({insufficient_neighbors/len(pts)*100:.2f}%)")
    return np.stack(covariances)


def run_gicp(
    source_pcd,
    target_pcd,
    init_trans=np.eye(4),
    optimizer="least_squares",
    max_iter=20,
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

    for _ in range(max_iter):
        # 최근접 대응 찾기
        tree = KDTree(tgt_pts)
        dists, idxs = tree.query(src)
        corr_src = src
        corr_tgt = tgt_pts[idxs]
        corr_src_cov = src_covs
        corr_tgt_cov = tgt_covs[idxs]

        # SVD optimizer 처리
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

        # G-ICP Hessian & gradient 계산
        H = np.zeros((6, 6))
        g = np.zeros((6, 1))
        R_prev = T_total[:3, :3]
        skip_cnt = 0; total_cnt = 0
        for i, (p, q) in enumerate(zip(corr_src, corr_tgt)):
            C = corr_src_cov[i] + R_prev @ corr_tgt_cov[i] @ R_prev.T + np.eye(3) * 1e-6
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                skip_cnt += 1
                continue
            total_cnt += 1
            r = (q - p).reshape(3, 1)
            J = np.zeros((3, 6))
            J[:, :3] = -skew(p)
            J[:, 3:] = -np.eye(3)
            H += J.T @ C_inv @ J
            g += J.T @ C_inv @ r

        # update solve
        try:
            if optimizer == "lm":
                lm = 1e-4
                dx = -np.linalg.solve(H + lm * np.eye(6), g)
            else:
                dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("[WARN] Hessian singular")
            break

        delta = dx.flatten()
        R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
        t_delta = delta[3:]
        T_delta = np.eye(4); T_delta[:3, :3] = R_delta; T_delta[:3, 3] = t_delta

        # 상태 업데이트
        src = (R_delta @ src.T).T + t_delta
        T_total = T_delta @ T_total

        # rmse_i 계산 및 최적값 갱신 (inliers 로직 변경 없음)
        d = np.linalg.norm(src - corr_tgt, axis=1)
        rmse_i = np.sqrt(np.mean(d[d < 2.0]**2)) if np.any(d < 2.0) else float('inf')
        if rmse_i < best_rmse:
            best_rmse = rmse_i
            best_T    = T_total.copy()

        if np.linalg.norm(delta) < tol:
            break

    # --- 최적 이터레이션으로 롤백 ---
    T_total = best_T.copy()

    # --- 최적 이터레이션으로 롤백 후 fresh 평가 ---
    final_src = (T_total[:3, :3] @ src_pts.T).T + T_total[:3, 3]
    tree = KDTree(tgt_pts)
    dists, idxs = tree.query(final_src)
    final_tgt = tgt_pts[idxs]
    d = np.linalg.norm(final_src - final_tgt, axis=1)
    inliers = d < 2.0
    fitness = np.sum(inliers) / len(d)
    rmse = np.sqrt(np.mean(d[inliers]**2)) if np.any(inliers) else float('inf')

    return T_total, fitness, rmse

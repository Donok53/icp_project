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
def run_p2p_icp(
    source_pcd,
    target_pcd,
    init_trans=np.eye(4),
    max_iter=20,
    tol=1e-6,
    optimizer='svd',       # 'svd', 'gauss_newton', 'lm'
    lambda_=1e-3           # LM initial damping
):
    # --- 준비: 포인트와 초기 변환 ---
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)
    T_total = init_trans.copy()
    best_T  = T_total.copy()
    best_rmse = float('inf')

    # 초기 source 위치
    src_trans = (T_total[:3,:3] @ src_pts.T).T + T_total[:3,3]

    # LM 전용 λ 초기화
    if optimizer == 'lm':
        lam = lambda_

    if optimizer == 'svd':
        # SVD-based ICP
        for _ in range(max_iter):
            tree    = KDTree(tgt_pts)
            dists, idxs = tree.query(src_trans)
            tgt_corr    = tgt_pts[idxs]

            # SVD 한 스텝
            T_delta = compute_transformation_svd(src_trans, tgt_corr)
            R_d     = T_delta[:3,:3]
            t_d     = T_delta[:3,3]
            src_trans = (R_d @ src_trans.T).T + t_d
            T_total   = T_delta @ T_total

            # RMSE 갱신
            d = np.linalg.norm(src_trans - tgt_corr, axis=1)
            mask = d < 2.0
            rmse_i = np.sqrt(np.mean(d[mask]**2)) if np.any(mask) else float('inf')
            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_T    = T_total.copy()

            if np.linalg.norm(t_d) < tol:
                break

    else:
        # 반복 최적화 (GN 또는 LM)
        x = np.zeros(6)  # 6-DoF 파라미터(회전축*각 + 평행이동)

        for iter_idx in range(max_iter):
            # 현재 파라미터 x → R_curr, t_curr
            omega = x[:3]
            theta = np.linalg.norm(omega)
            if theta < 1e-12:
                R_curr = np.eye(3)
            else:
                k = omega/theta
                K = skew(k)
                R_curr = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
            t_curr = x[3:]

            # 소스 점 변환 및 대응 찾기
            src_trans = (R_curr @ src_pts.T).T + t_curr
            tree = KDTree(tgt_pts)
            dists, idxs = tree.query(src_trans)
            mask_corr  = dists < np.inf
            P_corr = src_trans[mask_corr]
            Q_corr = tgt_pts[idxs[mask_corr]]

            # Hessian H, gradient g
            H = np.zeros((6,6))
            g = np.zeros(6)
            for p_i, q_i in zip(P_corr, Q_corr):
                r    = p_i - q_i
                J_i  = np.zeros((3,6))
                J_i[:, :3] = -skew(p_i)
                J_i[:, 3:] = -np.eye(3)
                H += J_i.T @ J_i
                g += J_i.T @ r

            # Levenberg–Marquardt 분기
            if optimizer == 'lm':
                # 비용 함수 정의
                def compute_cost(x_vec):
                    # x_vec → Rm, tm
                    om = x_vec[:3]; th = np.linalg.norm(om)
                    if th < 1e-12:
                        Rm = np.eye(3)
                    else:
                        kk = om/th; KK = skew(kk)
                        Rm = np.eye(3) + np.sin(th)*KK + (1-np.cos(th))*(KK@KK)
                    tm = x_vec[3:]
                    pts = (Rm @ src_pts.T).T + tm
                    d, _ = KDTree(tgt_pts).query(pts)
                    return np.mean(d**2)

                old_cost = compute_cost(x)
                # λ 스케줄링 루프
                while True:
                    try:
                        H_lm = H + lam * np.eye(6)
                        dx   = -np.linalg.solve(H_lm, g)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue

                    new_cost = compute_cost(x + dx)
                    if new_cost < old_cost:
                        # 개선되면 수용 & λ 줄이기
                        x  += dx
                        lam *= 0.1
                        break
                    else:
                        # 악화되면 λ 키우고 재시도
                        lam *= 10
                        if lam > 1e12:
                            x += dx
                            break

            else:
                # Gauss–Newton
                try:
                    dx = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    print("[WARN] Hessian singular")
                    break
                x += dx

            # 수렴 검사
            if np.linalg.norm(dx) < tol:
                print(f"[INFO] Converged after {iter_idx+1} iters")
                break

            # 파라미터 x → T_delta 누적
            dom = dx[:3]; th = np.linalg.norm(dom)
            if th < 1e-12:
                Rd = np.eye(3)
            else:
                kk = dom/th; KK = skew(kk)
                Rd = np.eye(3) + np.sin(th)*KK + (1-np.cos(th))*(KK@KK)
            td = dx[3:]
            T_delta = np.eye(4); T_delta[:3,:3]=Rd; T_delta[:3,3]=td
            T_total = T_delta @ T_total

            # RMSE 갱신
            iter_src = (T_total[:3,:3] @ src_pts.T).T + T_total[:3,3]
            d2, idx2  = KDTree(tgt_pts).query(iter_src)
            mask2     = d2 < 2.0
            rmse_i    = np.sqrt(np.mean(d2[mask2]**2)) if np.any(mask2) else float('inf')
            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_T    = T_total.copy()

    # 최적 이터레이션으로 롤백
    T_total = best_T.copy()

    # 최종 평가
    final_src = (T_total[:3,:3] @ src_pts.T).T + T_total[:3,3]
    treef     = KDTree(tgt_pts)
    d_final,_ = treef.query(final_src)
    mask_f    = d_final < 2.0
    fitness   = np.sum(mask_f)/len(d_final)
    rmse      = np.sqrt(np.mean(d_final[mask_f]**2)) if np.any(mask_f) else float('inf')

    return T_total, fitness, rmse

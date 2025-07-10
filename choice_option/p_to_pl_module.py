import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from choice_option.p_to_p_custom import compute_transformation_svd


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


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
    source_pcd,
    target_pcd,
    init_trans=np.eye(4),
    optimizer="svd",       # 'svd', 'gauss_newton', 'lm'
    max_iter=20,
    tol=1e-6,
    lambda_=1e-3           # LM 초기 λ
):
    source_pts    = np.asarray(source_pcd.points)
    target_pts    = np.asarray(target_pcd.points)
    target_normals = compute_normals(target_pts)

    T_total    = init_trans.copy()
    best_T     = T_total.copy()
    best_rmse  = float('inf')

    # 초기 source 위치
    src = (T_total[:3,:3] @ source_pts.T).T + T_total[:3,3]
    tree = KDTree(target_pts)

    # LM 전용 λ 초기화
    if optimizer == "lm":
        lam = lambda_

    for iter_idx in range(max_iter):
        # correspondences
        dists, idxs     = tree.query(src)
        corr_src        = src
        corr_tgt        = target_pts[idxs]
        corr_normals    = target_normals[idxs]

        # SVD branch
        if optimizer == "svd":
            diffs   = corr_src - corr_tgt
            dists_n = np.sum(diffs * corr_normals, axis=1, keepdims=True)
            proj_q  = corr_tgt + corr_normals * dists_n

            T_delta = compute_transformation_svd(corr_src, proj_q)
            R_delta = T_delta[:3,:3]
            t_delta = T_delta[:3,3]

            src     = (R_delta @ src.T).T + t_delta
            T_total = T_delta @ T_total

            # RMSE update
            final_corr = np.cross((src - corr_tgt), corr_normals)
            d = np.linalg.norm(final_corr, axis=1)
            mask = d < 2.0
            rmse_i = np.sqrt(np.mean(d[mask]**2)) if np.any(mask) else float('inf')
            if rmse_i < best_rmse:
                best_rmse = rmse_i
                best_T    = T_total.copy()

            if np.linalg.norm(t_delta) < tol:
                break
            else:
                continue

        # LS/GN/LM branch: build H, b
        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        for p, q, n in zip(corr_src, corr_tgt, corr_normals):
            Δ = p - q
            r = Δ.dot(n)
            J = np.zeros((1, 6))
            J[0, :3] = -(Δ @ skew(n))
            J[0, 3:] = n
            H += J.T @ J
            b += J.T * r

        # LM scheduling
        if optimizer == "lm":
            def compute_cost(T):
                pts = (T[:3,:3] @ source_pts.T).T + T[:3,3]
                d, _ = KDTree(target_pts).query(pts)
                return np.mean(d**2)

            old_cost = compute_cost(T_total)
            while True:
                try:
                    H_lm = H + lam * np.eye(6)
                    delta = -np.linalg.solve(H_lm, b)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue

                # candidate transform
                angle = delta[:3].flatten()
                trans = delta[3:].flatten()
                R_cand = o3d.geometry.get_rotation_matrix_from_axis_angle(angle)
                T_cand = np.eye(4)
                T_cand[:3,:3] = R_cand
                T_cand[:3,3]  = trans
                T_new  = T_cand @ T_total

                new_cost = compute_cost(T_new)
                if new_cost < old_cost:
                    # accept
                    T_total = T_new
                    src     = (R_cand @ src.T).T + trans
                    lam *= 0.1
                    break
                else:
                    lam *= 10
                    if lam > 1e12:
                        T_total = T_new
                        src     = (R_cand @ src.T).T + trans
                        break

        else:
            # Gauss-Newton
            try:
                delta = -np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                print("[WARN] Singular matrix during optimization.")
                break
            # step size limiter
            max_step = 0.1
            norm_delta = np.linalg.norm(delta)
            if norm_delta > max_step:
                delta *= (max_step / norm_delta)

            angle = delta[:3].flatten()
            trans = delta[3:].flatten()
            R_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(angle)
            T_delta = np.eye(4)
            T_delta[:3,:3] = R_delta
            T_delta[:3,3]  = trans
            src     = (R_delta @ src.T).T + trans
            T_total = T_delta @ T_total

        # RMSE update
        final_corr = np.cross((src - corr_tgt), corr_normals)
        d = np.linalg.norm(final_corr, axis=1)
        mask = d < 2.0
        rmse_i = np.sqrt(np.mean(d[mask]**2)) if np.any(mask) else float('inf')
        if rmse_i < best_rmse:
            best_rmse = rmse_i
            best_T    = T_total.copy()

        # convergence check
        if np.linalg.norm(delta) < tol:
            break

    # rollback
    T_total = best_T.copy()

    # final evaluation
    src = (T_total[:3,:3] @ source_pts.T).T + T_total[:3,3]
    dists, idxs = tree.query(src)
    corr_q = target_pts[idxs]
    corr_v = target_normals[idxs]
    final_corr = np.cross((src - corr_q), corr_v)
    d = np.linalg.norm(final_corr, axis=1)
    inliers = d < 2.0
    fitness = np.sum(inliers) / len(d)
    rmse = np.sqrt(np.mean(d[inliers]**2)) if np.any(inliers) else float('inf')

    return T_total, fitness, rmse
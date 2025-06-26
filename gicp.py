import open3d as o3d
import numpy as np
import os
from copy import deepcopy
from scipy.linalg import inv
from scipy.spatial.transform import Rotation as R

# ──────────────────────── 설정 ────────────────────────
frame_gap = 5  # 프레임 간격
max_iterations = 10
fitness_threshold = 0.001

# ──────────────────────── 유틸 함수 ────────────────────────
def load_bin_as_pcd(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def load_poses_kitti(gt_path):
    poses = []
    with open(gt_path, 'r') as f:
        for line in f:
            values = np.fromstring(line.strip(), sep=' ')
            if len(values) == 13:
                values = values[1:]  # 첫 index 제거
            if len(values) != 12:
                continue
            T = values.reshape(3, 4)
            T_homo = np.eye(4)
            T_homo[:3, :] = T
            poses.append(T_homo)
    return poses

def compute_ate_rmse(pred_poses, gt_poses):
    errors = []
    for T_pred, T_gt in zip(pred_poses, gt_poses):
        trans_error = T_pred[:3, 3] - T_gt[:3, 3]
        errors.append(np.linalg.norm(trans_error))
    return np.sqrt(np.mean(np.square(errors)))

def compute_covariances(pcd, k=20):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    covariances = []
    points = np.asarray(pcd.points)

    for i in range(len(points)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx, :]
        if len(neighbors) < 5:
            covariances.append(np.eye(3) * 1e-6)
            continue
        mean = np.mean(neighbors, axis=0)
        cov = np.cov((neighbors - mean).T)
        cov += np.eye(3) * 1e-6
        covariances.append(cov)

    return covariances

def gicp(source, target, max_iterations):
    source_covs = compute_covariances(source)
    target_covs = compute_covariances(target)

    source_tree = o3d.geometry.KDTreeFlann(target)
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    R_curr = np.eye(3)
    t_curr = np.zeros(3)

    for _ in range(max_iterations):
        A = np.zeros((6, 6))
        b = np.zeros(6)
        valid_correspondences = 0

        for i in range(len(source_points)):
            p = source_points[i]
            p_cov = source_covs[i]
            p_trans = R_curr @ p + t_curr

            _, idx, _ = source_tree.search_knn_vector_3d(p_trans, 1)
            if not idx:
                continue

            q = target_points[idx[0]]
            q_cov = target_covs[idx[0]]

            C = p_cov + R_curr @ q_cov @ R_curr.T
            try:
                C_inv = inv(C)
            except np.linalg.LinAlgError:
                continue

            r = p_trans - q
            J = np.zeros((3, 6))
            J[:, :3] = -np.eye(3)
            J[:, 3:] = -np.array([
                [0, -p_trans[2], p_trans[1]],
                [p_trans[2], 0, -p_trans[0]],
                [-p_trans[1], p_trans[0], 0]
            ])

            A += J.T @ C_inv @ J
            b += J.T @ C_inv @ r
            valid_correspondences += 1

        if valid_correspondences < 10:
            return None, None, 0.0, float('inf')

        delta = np.linalg.solve(A, -b)
        delta_t = delta[:3]
        delta_angle = delta[3:]

        angle = np.linalg.norm(delta_angle)
        axis = delta_angle / angle if angle > 1e-6 else np.zeros(3)
        R_update = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

        R_curr = R_update @ R_curr
        t_curr = R_update @ t_curr + delta_t

    # Fitness 계산 (inlier 비율)
    inlier_count = 0
    src_pts = (R_curr @ source_points.T).T + t_curr
    for p in src_pts:
        _, idx, _ = source_tree.search_knn_vector_3d(p, 1)
        q = target_points[idx[0]]
        if np.linalg.norm(p - q) < 1.5:
            inlier_count += 1

    fitness = inlier_count / len(source_points)
    rmse = np.sqrt(np.mean([np.linalg.norm(src_pts[i] - target_points[source_tree.search_knn_vector_3d(src_pts[i], 1)[1][0]])**2 for i in range(len(src_pts))]))

    T_final = np.eye(4)
    T_final[:3, :3] = R_curr
    T_final[:3, 3] = t_curr
    return T_final, R_curr, fitness, rmse

# ──────────────────────── 메인 파이프라인 ────────────────────────
def run_gicp_pipeline(bin_dir, gt_pose_path):
    bin_files_all = sorted([f for f in os.listdir(bin_dir) if f.endswith('.bin')])
    bin_files = bin_files_all[::frame_gap]

    gt_all = load_poses_kitti(gt_pose_path)
    gt_poses = gt_all[::frame_gap]

    pred_poses = [np.eye(4)]
    global_map = o3d.geometry.PointCloud()

    target = load_bin_as_pcd(os.path.join(bin_dir, bin_files[0]))
    global_map += target

    for i in range(1, len(bin_files)):
        curr_name = bin_files[i - 1].replace('.bin', '')
        next_name = bin_files[i].replace('.bin', '')
        print(f"[INFO] Aligning frame {curr_name} → {next_name}")

        source = load_bin_as_pcd(os.path.join(bin_dir, bin_files[i]))

        T_icp, _, fitness, rmse = gicp(source, target, max_iterations)
        print(f"[ICP] Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")

        if T_icp is not None and fitness > fitness_threshold:
            T_global = pred_poses[-1] @ T_icp
            pred_poses.append(T_global)
            source.transform(T_global)
            global_map += source
            target = deepcopy(source)
        else:
            print(f"[WARN] Fitness too low. Skipping frame {next_name}")
            pred_poses.append(pred_poses[-1])

    rmse = compute_ate_rmse(pred_poses[:len(gt_poses)], gt_poses)
    print(f"[INFO] ATE 평가 시작...")
    print(f"[EVAL] ATE RMSE (GICP): {rmse:.4f} meters")

    with open("trajectory_gicp.txt", "w") as f:
        for i, T in enumerate(pred_poses):
            t = T[:3, 3]
            quat = R.from_matrix(T[:3, :3]).as_quat()
            frame_id = bin_files[i].replace('.bin', '')
            f.write(f"{frame_id} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")

    print("[INFO] 전체 맵 시각화 중...")
    o3d.visualization.draw_geometries([global_map])

# ──────────────────────── 실행 예시 ────────────────────────
if __name__ == "__main__":
    bin_dir = "D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0004_sync/velodyne_points/data"
    gt_pose_path = "D:/kitti360/data_poses/2013_05_28_drive_0004_sync/poses.txt"
    run_gicp_pipeline(bin_dir, gt_pose_path)

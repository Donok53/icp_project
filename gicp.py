import open3d as o3d
import numpy as np
import os
from copy import deepcopy
from scipy.linalg import inv

# ───────────── GICP 함수 ─────────────
def compute_covariances(pcd, k=20):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    covariances = []
    points = np.asarray(pcd.points)

    for i in range(len(points)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx, :]
        mean = np.mean(neighbors, axis=0)
        cov = np.cov((neighbors - mean).T)
        cov += np.eye(3) * 1e-6
        covariances.append(cov)

    return covariances

def gicp(source, target, max_iterations=10):
    source_covs = compute_covariances(source)
    target_covs = compute_covariances(target)

    source_tree = o3d.geometry.KDTreeFlann(target)
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    R = np.eye(3)
    t = np.zeros(3)

    for iter in range(max_iterations):
        A = np.zeros((6, 6))
        b = np.zeros(6)

        for i in range(len(source_points)):
            p = source_points[i]
            p_cov = source_covs[i]
            p_trans = R @ p + t

            _, idx, _ = source_tree.search_knn_vector_3d(p_trans, 1)
            q = target_points[idx[0]]
            q_cov = target_covs[idx[0]]

            C = p_cov + R @ q_cov @ R.T
            C_inv = inv(C)
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

        delta = np.linalg.solve(A, -b)
        delta_t = delta[:3]
        delta_angle = delta[3:]

        angle = np.linalg.norm(delta_angle)
        if angle < 1e-6:
            R_update = np.eye(3)
        else:
            axis = delta_angle / angle
            R_update = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

        R = R_update @ R
        t = R_update @ t + delta_t

    return R, t

# ───────────── 유틸 함수들 ─────────────
def load_bin_as_pcd(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def load_poses_kitti(gt_path):
    poses = []
    with open(gt_path, 'r') as f:
        for i, line in enumerate(f):
            values = np.fromstring(line.strip(), sep=' ')
            if len(values) == 13:
                values = values[1:]  # 첫 번째 index 값 제거
            if len(values) != 12:
                print(f"[WARN] Line {i+1} has {len(values)} elements. Skipping.")
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

# ───────────── 메인 파이프라인 ─────────────
def run_gicp_pipeline(bin_dir, gt_pose_path):
    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith('.bin')])
    gt_poses = load_poses_kitti(gt_pose_path)
    pred_poses = [np.eye(4)]

    target = load_bin_as_pcd(os.path.join(bin_dir, bin_files[0]))

    for i in range(1, len(gt_poses)):
        print(f"[INFO] Aligning frame {i-1} → {i}")
        source = load_bin_as_pcd(os.path.join(bin_dir, bin_files[i]))

        try:
            R_est, t_est = gicp(source, target)
        except np.linalg.LinAlgError:
            print(f"[WARN] Frame {i} 정합 실패, 이전 포즈 복사")
            pred_poses.append(pred_poses[-1])
            target = deepcopy(source)
            continue

        T_delta = np.eye(4)
        T_delta[:3, :3] = R_est
        T_delta[:3, 3] = t_est
        T_i = pred_poses[-1] @ T_delta
        pred_poses.append(T_i)

        target = deepcopy(source)

    rmse = compute_ate_rmse(pred_poses[:len(gt_poses)], gt_poses)
    print(f"[EVAL] ATE RMSE: {rmse:.4f} meters")

    return pred_poses

# ───────────── 실행 예시 ─────────────
if __name__ == "__main__":
    bin_dir = "D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0004_sync/velodyne_points/data"
    gt_pose_path = "D:/kitti360/data_poses/2013_05_28_drive_0004_sync/poses.txt"
    run_gicp_pipeline(bin_dir, gt_pose_path)




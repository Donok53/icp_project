import numpy as np
import open3d as o3d
import os
from copy import deepcopy
from scipy.linalg import inv
from scipy.spatial.transform import Rotation as R


def load_bin_as_pcd(path, voxel_size=0.2):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pcd.voxel_down_sample(voxel_size)


def load_poses_kitti(gt_path):
    poses = []
    with open(gt_path, "r") as f:
        for line in f:
            values = np.fromstring(line.strip(), sep=" ")
            if len(values) == 13:
                values = values[1:]
            if len(values) != 12:
                continue
            T = values.reshape(3, 4)
            T_homo = np.eye(4)
            T_homo[:3, :] = T
            poses.append(T_homo)
    return poses


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


def gicp(source, target, init_R=np.eye(3), init_t=np.zeros(3), max_iterations=20):
    source_covs = compute_covariances(source)
    target_covs = compute_covariances(target)
    target_tree = o3d.geometry.KDTreeFlann(target)

    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    R_curr = init_R
    t_curr = init_t

    for _ in range(max_iterations):
        A = np.zeros((6, 6))
        b = np.zeros(6)

        for i in range(len(source_points)):
            p = source_points[i]
            p_cov = source_covs[i]
            p_trans = R_curr @ p + t_curr

            _, idx, _ = target_tree.search_knn_vector_3d(p_trans, 1)
            q = target_points[idx[0]]
            q_cov = target_covs[idx[0]]

            C = p_cov + R_curr @ q_cov @ R_curr.T
            C_inv = inv(C)
            r = p_trans - q

            J = np.zeros((3, 6))
            J[:, :3] = -np.eye(3)
            J[:, 3:] = -np.array(
                [
                    [0, -p_trans[2], p_trans[1]],
                    [p_trans[2], 0, -p_trans[0]],
                    [-p_trans[1], p_trans[0], 0],
                ]
            )

            A += J.T @ C_inv @ J
            b += J.T @ C_inv @ r

        delta = np.linalg.solve(A, -b)
        delta_t = delta[:3]
        delta_angle = delta[3:]
        angle = np.linalg.norm(delta_angle)

        R_update = (
            np.eye(3)
            if angle < 1e-6
            else o3d.geometry.get_rotation_matrix_from_axis_angle(delta_angle)
        )
        R_curr = R_update @ R_curr
        t_curr = R_update @ t_curr + delta_t

    T = np.eye(4)
    T[:3, :3] = R_curr
    T[:3, 3] = t_curr
    return T


def compute_ate_rmse(pred_poses, gt_poses):
    errors = []
    for T_pred, T_gt in zip(pred_poses, gt_poses):
        trans_error = T_pred[:3, 3] - T_gt[:3, 3]
        errors.append(np.linalg.norm(trans_error))
    return np.sqrt(np.mean(np.square(errors)))


def run_gicp_pipeline(bin_dir, gt_pose_path, frame_gap=5):
    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    gt_poses = load_poses_kitti(gt_pose_path)
    pred_poses = [np.eye(4)]

    frame_ids = list(range(0, min(len(bin_files), len(gt_poses)), frame_gap))
    target = load_bin_as_pcd(os.path.join(bin_dir, bin_files[frame_ids[0]]))

    for i in range(1, len(frame_ids)):
        src_id = frame_ids[i]
        tgt_id = frame_ids[i - 1]

        print(f"[INFO] Aligning frame {tgt_id} → {src_id}")
        source = load_bin_as_pcd(os.path.join(bin_dir, bin_files[src_id]))
        target = load_bin_as_pcd(os.path.join(bin_dir, bin_files[tgt_id]))

        T_init = np.linalg.inv(gt_poses[tgt_id]) @ gt_poses[src_id]
        T_icp = gicp(source, target, T_init[:3, :3], T_init[:3, 3])
        T_i = pred_poses[-1] @ T_icp
        pred_poses.append(T_i)

    rmse = compute_ate_rmse(pred_poses, [gt_poses[i] for i in frame_ids])
    print(f"[EVAL] ATE RMSE (GICP): {rmse:.4f} meters")

    with open("trajectory_gicp.txt", "w") as f:
        for fid, T in zip(frame_ids, pred_poses):
            t = T[:3, 3]
            quat = R.from_matrix(T[:3, :3]).as_quat()
            f.write(
                f"{fid} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n"
            )

    return rmse, pred_poses


# ───────────── 실행 예시 ─────────────
run_gicp_pipeline(
    bin_dir="D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0004_sync/velodyne_points/data",
    gt_pose_path="D:/kitti360/data_poses/2013_05_28_drive_0004_sync/poses.txt",
)

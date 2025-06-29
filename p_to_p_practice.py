import open3d as o3d
import numpy as np
from scipy.spatial import KDTree


def load_point_cloud(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def icp_point_to_point(src_pts, tgt_pts, max_iter=20, tol=1e-6):
    src = src_pts.copy()
    tgt = tgt_pts.copy()
    prev_error = float("inf")

    for _ in range(max_iter):
        tree = KDTree(tgt)
        distances, indices = tree.query(src)
        tgt_matched = tgt[indices]

        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt_matched, axis=0)
        src_centered = src - centroid_src
        tgt_centered = tgt_matched - centroid_tgt

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src

        src = (R @ src.T).T + t
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tol:
            break
        prev_error = mean_error

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_gt_poses_with_indices(file_path):
    gt_poses = []
    frame_ids = []
    with open(file_path, "r") as f:
        for line in f:
            values = np.fromstring(line.strip(), sep=" ")
            if len(values) != 13:
                continue
            frame_id = int(values[0])
            pose_vec = values[1:]
            T = np.eye(4)
            T[:3, :4] = pose_vec.reshape(3, 4)
            frame_ids.append(frame_id)
            gt_poses.append(T)
    return frame_ids, gt_poses


def compute_ate(estimated_poses, gt_poses):
    errors = []
    for est, gt in zip(estimated_poses, gt_poses):
        trans_est = est[:3, 3]
        trans_gt = gt[:3, 3]
        error = np.linalg.norm(trans_est - trans_gt)
        errors.append(error)
    ate_rmse = np.sqrt(np.mean(np.square(errors)))
    return ate_rmse, errors


if __name__ == "__main__":
    base_dir = "D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data"
    gt_pose_path = "D:/kitti360/data_poses/2013_05_28_drive_0000_sync/poses.txt"

    # GT에서 frame id와 pose를 같이 불러옴
    frame_ids, gt_poses = load_gt_poses_with_indices(gt_pose_path, limit=200)

    frame_ids = frame_ids[:300]
    gt_poses = gt_poses[:300]

    pose = np.eye(4)
    global_map = o3d.geometry.PointCloud()
    trajectory_est = []

    prev_idx = frame_ids[0]
    prev_pcd = load_point_cloud(f"{base_dir}/{prev_idx:010d}.bin")
    prev_pts = np.asarray(prev_pcd.points)
    prev_pcd.transform(pose)
    global_map += prev_pcd
    trajectory_est.append(pose.copy())

    for i in range(1, len(frame_ids)):
        curr_idx = frame_ids[i]
        print(f"[INFO] Aligning frame {curr_idx} to {frame_ids[i-1]}")

        curr_pcd = load_point_cloud(f"{base_dir}/{curr_idx:010d}.bin")
        curr_pts = np.asarray(curr_pcd.points)

        T = icp_point_to_point(prev_pts, curr_pts)
        pose = pose @ T
        trajectory_est.append(pose.copy())

        aligned_pcd = load_point_cloud(f"{base_dir}/{curr_idx:010d}.bin")
        aligned_pcd.transform(pose)
        global_map += aligned_pcd

        prev_pts = curr_pts.copy()

    # ATE 평가
    print("[INFO] ATE 평가 시작")
    ate_rmse, errors = compute_ate(trajectory_est, gt_poses)
    print(f"[EVAL] ATE RMSE: {ate_rmse:.4f} meters")

    print("[INFO] 정합 완료. 시각화 시작.")
    o3d.visualization.draw_geometries([global_map])

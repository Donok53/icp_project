import open3d as o3d
import numpy as np
import os
from scipy.spatial import KDTree

def load_point_cloud(file_path, voxel_size=0.1):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] {file_path} not found.")
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size)

def estimate_normals(pcd, radius=1.5, max_nn=50):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals)

def run_icp_point_to_plane(source, target, init_trans=np.eye(4), threshold=1.5):
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50))
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg.transformation, reg.fitness, reg.inlier_rmse

def load_gt_poses_with_indices(file_path):
    gt_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = np.fromstring(line.strip(), sep=' ')
            if len(values) != 13: continue
            frame_id = int(values[0])
            T = np.eye(4)
            T[:3, :4] = values[1:].reshape(3, 4)
            gt_dict[frame_id] = T
    return gt_dict

def compute_ate_relative(trajectory_est, gt_dict):
    errors = []
    T0_gt_inv = np.linalg.inv(gt_dict[trajectory_est[0][0]])
    T0_est_inv = np.linalg.inv(trajectory_est[0][1])
    for fid, est_pose in trajectory_est:
        if fid not in gt_dict: continue
        rel_gt = T0_gt_inv @ gt_dict[fid]
        rel_est = T0_est_inv @ est_pose
        errors.append(np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3]))
    return np.sqrt(np.mean(np.square(errors)))

def get_available_frame_ids(base_dir):
    return sorted([int(f.replace('.bin', '')) for f in os.listdir(base_dir) if f.endswith('.bin')])

if __name__ == "__main__":
    base_dir = "D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0004_sync/velodyne_points/data"
    gt_pose_path = "D:/kitti360/data_poses/2013_05_28_drive_0004_sync/poses.txt"

    frame_ids = get_available_frame_ids(base_dir)
    gt_dict = load_gt_poses_with_indices(gt_pose_path)
    start_idx = max(frame_ids[0], min(gt_dict.keys()))
    end_idx = start_idx + 200
    frame_gap = 5

    valid_ids = [fid for fid in range(start_idx, end_idx, frame_gap)
                 if fid in gt_dict and os.path.exists(f"{base_dir}/{fid:010d}.bin")]

    pose = np.eye(4)
    global_map = o3d.geometry.PointCloud()
    trajectory_est = []

    prev_id = valid_ids[0]
    prev_pcd = load_point_cloud(f"{base_dir}/{prev_id:010d}.bin")
    prev_pcd.transform(pose)
    global_map += prev_pcd
    trajectory_est.append((prev_id, pose.copy()))

    for curr_id in valid_ids[1:]:
        print(f"[INFO] Aligning frame {curr_id}")
        curr_pcd = load_point_cloud(f"{base_dir}/{curr_id:010d}.bin")

        T_gt_prev = gt_dict[prev_id]
        T_gt_curr = gt_dict[curr_id]
        T_init = np.linalg.inv(T_gt_prev) @ T_gt_curr

        T_icp, fitness, rmse = run_icp_point_to_plane(curr_pcd, prev_pcd, init_trans=T_init)
        print(f"[ICP] Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")

        if fitness > 0.9:
            pose = pose @ T_icp
            curr_pcd.transform(pose)
            global_map += curr_pcd
            trajectory_est.append((curr_id, pose.copy()))
            prev_pcd = curr_pcd
            prev_id = curr_id
        else:
            print(f"[WARN] Fitness too low. Skipping frame {curr_id}")

    print("[INFO] ATE 평가 시작...")
    ate_rmse = compute_ate_relative(trajectory_est, gt_dict)
    print(f"[EVAL] ATE RMSE: {ate_rmse:.4f} meters")

    print("[INFO] 전체 맵 정합 완료. 시각화 중...")
    o3d.visualization.draw_geometries([global_map])



import open3d as o3d
import numpy as np
import os

def load_point_cloud(file_path, voxel_size=0.2):
    """KITTI 포맷 .bin 파일 불러오기 + 다운샘플링"""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size)

def run_icp_open3d(source, target, threshold=1.0):
    """Open3D ICP로 정합"""
    trans_init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg.transformation, reg.fitness, reg.inlier_rmse

def load_gt_poses_with_indices(file_path):
    """GT pose 파일을 frame_id 기반 딕셔너리로 불러옴"""
    gt_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = np.fromstring(line.strip(), sep=' ')
            if len(values) != 13:
                continue
            frame_id = int(values[0])
            pose_vec = values[1:]
            T = np.eye(4)
            T[:3, :4] = pose_vec.reshape(3, 4)
            gt_dict[frame_id] = T
    return gt_dict

def compute_ate_relative(trajectory_est, gt_dict):
    """추정 trajectory와 GT를 상대 pose 기준으로 비교"""
    errors = []

    # GT가 존재하는 frame만 필터링
    filtered = [(fid, pose) for fid, pose in trajectory_est if fid in gt_dict]
    if len(filtered) < 2:
        print("[ERROR] GT와 매칭되는 프레임 수 부족")
        return float('nan')

    T0_est_inv = np.linalg.inv(filtered[0][1])
    T0_gt_inv = np.linalg.inv(gt_dict[filtered[0][0]])

    for fid, T_est in filtered:
        T_gt = gt_dict[fid]
        rel_est = T0_est_inv @ T_est
        rel_gt = T0_gt_inv @ T_gt
        error = np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3])
        errors.append(error)

    ate_rmse = np.sqrt(np.mean(np.square(errors)))
    return ate_rmse


if __name__ == "__main__":
    base_dir = "D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0002_sync/velodyne_points/data"
    gt_pose_path = "D:/kitti360/data_poses/2013_05_28_drive_0002_sync/poses.txt"

    start_idx = 0
    end_idx = 90        # 프레임 개수 조절 가능
    frame_gap = 2        # 2프레임 간격
    threshold = 1.0

    pose = np.eye(4)
    global_map = o3d.geometry.PointCloud()
    trajectory_est = []

    prev_pcd = load_point_cloud(f"{base_dir}/{start_idx:010d}.bin")
    prev_pcd.transform(pose)
    global_map += prev_pcd
    trajectory_est.append((start_idx, pose.copy()))

    for i in range(start_idx + frame_gap, end_idx, frame_gap):
        print(f"[INFO] Aligning frame {i}")
        curr_pcd = load_point_cloud(f"{base_dir}/{i:010d}.bin")

        T, fitness, rmse = run_icp_open3d(curr_pcd, prev_pcd, threshold)
        print(f"[ICP] Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")

        pose = pose @ np.linalg.inv(T)
        curr_pcd.transform(pose)
        global_map += curr_pcd
        trajectory_est.append((i, pose.copy()))

        prev_pcd = curr_pcd

    print("[INFO] ATE 평가 시작...")
    gt_dict = load_gt_poses_with_indices(gt_pose_path)
    ate_rmse = compute_ate_relative(trajectory_est, gt_dict)
    print(f"[EVAL] ATE RMSE: {ate_rmse:.4f} meters")

    print("[INFO] 전체 맵 정합 완료. 시각화 중...")
    o3d.visualization.draw_geometries([global_map])



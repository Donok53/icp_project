import open3d as o3d
import numpy as np
import os

def load_point_cloud(file_path, voxel_size=0.2):
    """KITTI 포맷 .bin 파일 로드 및 다운샘플"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] {file_path} not found.")
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size)

def run_icp_point_to_line(source, target, init_trans=np.eye(4), threshold=1.0):
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg.transformation, reg.fitness, reg.inlier_rmse

def load_gt_poses_with_indices(file_path):
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
    errors = []
    start_id = trajectory_est[0][0]
    if start_id not in gt_dict:
        raise ValueError(f"[ERROR] GT pose for frame {start_id} not found.")
    T0_gt_inv = np.linalg.inv(gt_dict[start_id])
    T0_est_inv = np.linalg.inv(trajectory_est[0][1])

    for frame_id, est_pose in trajectory_est:
        if frame_id not in gt_dict:
            continue
        T_gt = gt_dict[frame_id]
        rel_gt = T0_gt_inv @ T_gt
        rel_est = T0_est_inv @ est_pose
        trans_error = rel_est[:3, 3] - rel_gt[:3, 3]
        errors.append(np.linalg.norm(trans_error))
    return np.sqrt(np.mean(np.square(errors)))

def get_available_frame_ids(base_dir):
    files = os.listdir(base_dir)
    frame_ids = sorted([
        int(f.replace('.bin', '')) for f in files if f.endswith('.bin')
    ])
    return frame_ids

if __name__ == "__main__":
    base_dir = "/home/byeongjae/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0002_sync/velodyne_points/data"
    gt_pose_path = "/home/byeongjae/kitti360/data_poses/2013_05_28_drive_0002_sync/poses.txt"

    frame_ids = get_available_frame_ids(base_dir)
    print(f"[INFO] 사용할 수 있는 프레임 ID 범위: {frame_ids[0]} ~ {frame_ids[-1]}")

    gt_dict = load_gt_poses_with_indices(gt_pose_path)
    print(f"[INFO] GT pose 개수: {len(gt_dict)}")

    start_idx = max(frame_ids[0], min(gt_dict.keys()))
    end_idx = start_idx + 200
    frame_gap = 5
    threshold = 0.8

    valid_frame_ids = [fid for fid in range(start_idx, end_idx, frame_gap)
                       if fid in gt_dict and os.path.exists(f"{base_dir}/{fid:010d}.bin")]
    print(f"[INFO] 실제 정합 대상 프레임 수: {len(valid_frame_ids)}")

    pose = np.eye(4)
    global_map = o3d.geometry.PointCloud()
    trajectory_est = []

    first_frame = valid_frame_ids[0]
    prev_pcd = load_point_cloud(f"{base_dir}/{first_frame:010d}.bin")
    prev_pcd.transform(pose)
    global_map += prev_pcd
    trajectory_est.append((first_frame, pose.copy()))
    prev_frame_id = first_frame

    for i in valid_frame_ids[1:]:
        print(f"[INFO] Aligning frame {i}")
        curr_pcd = load_point_cloud(f"{base_dir}/{i:010d}.bin")

        # GT 기반 초기 추정 pose
        T_gt_prev = gt_dict[prev_frame_id]
        T_gt_curr = gt_dict[i]
        T_init = np.linalg.inv(T_gt_prev) @ T_gt_curr

        # ICP
        T_icp, fitness, rmse = run_icp_point_to_line(curr_pcd, prev_pcd, init_trans=T_init, threshold=threshold)
        print(f"[ICP] Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")

        if fitness > 0.9:  # ✅ 다시 적용
            pose = pose @ np.linalg.inv(T_icp)  # ✅ 역변환 누적 유지
            curr_pcd.transform(pose)
            global_map += curr_pcd
            trajectory_est.append((i, pose.copy()))
            prev_pcd = curr_pcd
            prev_frame_id = i
        else:
            print(f"[WARN] 정합 실패 (fitness={fitness:.2f}), pose 누적 생략")


    # ATE 평가
    print("[INFO] ATE 평가 시작...")
    ate_rmse = compute_ate_relative(trajectory_est, gt_dict)
    print(f"[EVAL] ATE RMSE: {ate_rmse:.4f} meters")

    print("[DEBUG] T_icp:")
    print(T_icp)

    print("[INFO] 전체 맵 정합 완료. 시각화 중...")
    o3d.visualization.draw_geometries([global_map])


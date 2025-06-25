import open3d as o3d
import numpy as np
import os
from scipy.spatial import KDTree

def load_point_cloud(file_path, voxel_size=0.2):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size)

def estimate_covariances(pcd, radius=1.0, max_nn=20):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    covariances = []
    pts = np.asarray(pcd.points)

    for i in range(len(pts)):
        _, idxs, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if len(idxs) < 5:
            covariances.append(np.eye(3) * 1e-3)
            continue
        neighbors = pts[idxs] - pts[i]
        cov = neighbors.T @ neighbors / (len(idxs) - 1)
        U, S, Vt = np.linalg.svd(cov)
        S = np.clip(S, 1e-6, None)
        cov = U @ np.diag(S) @ U.T
        covariances.append(cov)
    return covariances

def se3_hat(xi):
    omega = xi[:3]
    v = xi[3:]
    Omega = np.array([[0, -omega[2], omega[1]],
                      [omega[2], 0, -omega[0]],
                      [-omega[1], omega[0], 0]])
    hat = np.zeros((4, 4))
    hat[:3, :3] = Omega
    hat[:3, 3] = v
    return hat

def gicp_icp(src_pts, tgt_pts, src_covs, tgt_covs, init_T=np.eye(4), max_iter=20, tol=1e-4):
    T = init_T.copy()
    for _ in range(max_iter):
        src_transformed = (T[:3, :3] @ src_pts.T).T + T[:3, 3]
        tree = KDTree(tgt_pts)
        distances, indices = tree.query(src_transformed)
        tgt_corr = tgt_pts[indices]
        cov_combined = []

        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        for i in range(len(src_transformed)):
            p = src_transformed[i]
            q = tgt_corr[i]
            C = src_covs[i] + tgt_covs[indices[i]]
            C_inv = np.linalg.inv(C)

            r = (p - q).reshape(3, 1)
            J = np.zeros((3, 6))
            J[:, :3] = -np.eye(3)
            J[:, 3:] = -np.array([[0, -p[2], p[1]],
                                  [p[2], 0, -p[0]],
                                  [-p[1], p[0], 0]])
            H += J.T @ C_inv @ J
            b += J.T @ C_inv @ r

        try:
            dx = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            break

        xi = dx.flatten()
        T_update = np.eye(4)
        T_update[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(xi[3:])
        T_update[:3, 3] = xi[:3]
        T = T_update @ T

        if np.linalg.norm(dx) < tol:
            break
    return T

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
    base_dir = "/home/byeongjae/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0002_sync/velodyne_points/data"
    gt_pose_path = "/home/byeongjae/kitti360/data_poses/2013_05_28_drive_0002_sync/poses.txt"

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
    prev_pts = np.asarray(prev_pcd.points)
    prev_covs = estimate_covariances(prev_pcd)
    global_map += prev_pcd
    trajectory_est.append((prev_id, pose.copy()))

    for curr_id in valid_ids[1:]:
        print(f"[INFO] Aligning frame {curr_id}")
        curr_pcd = load_point_cloud(f"{base_dir}/{curr_id:010d}.bin")
        curr_pts = np.asarray(curr_pcd.points)
        curr_covs = estimate_covariances(curr_pcd)

        T_gt_prev = gt_dict[prev_id]
        T_gt_curr = gt_dict[curr_id]
        T_init = np.linalg.inv(T_gt_prev) @ T_gt_curr

        T_icp = gicp_icp(curr_pts, prev_pts, curr_covs, prev_covs, init_T=T_init)
        pose = pose @ T_icp

        curr_pcd.transform(pose)
        global_map += curr_pcd
        trajectory_est.append((curr_id, pose.copy()))

        prev_id = curr_id
        prev_pts = curr_pts
        prev_covs = curr_covs

    print("[INFO] ATE 평가 시작...")
    ate_rmse = compute_ate_relative(trajectory_est, gt_dict)
    print(f"[EVAL] ATE RMSE: {ate_rmse:.4f} meters")
    o3d.visualization.draw_geometries([global_map])

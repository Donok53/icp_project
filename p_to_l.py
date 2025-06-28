import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R


def load_point_cloud(file_path, voxel_size=0.2):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size)


def estimate_line_directions(target_pcd, k=10):
    points = np.asarray(target_pcd.points)
    tree = KDTree(points)
    directions = []
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k)
        neighbors = points[idx]
        cov = np.cov((neighbors - neighbors.mean(axis=0)).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        directions.append(direction)
    return np.stack(directions)


def point_to_line_residual(p_src, p_tgt, v):
    d = p_src - p_tgt
    proj = v @ d
    return d - proj * v


def icp_point_to_line_3d(source_pcd, target_pcd, max_iter=20):
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)
    tree = KDTree(tgt_pts)
    v_list = estimate_line_directions(target_pcd)

    T = np.eye(4)
    for _ in range(max_iter):
        src_h = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        src_trans = (T @ src_h.T).T[:, :3]

        residuals = []
        J_all = []

        for i, p in enumerate(src_trans):
            _, idx = tree.query(p)
            q = tgt_pts[idx]
            v = v_list[idx]
            r = point_to_line_residual(p, q, v)
            residuals.append(r)

            px, py, pz = src_pts[i]
            skew = np.array([[0, -pz, py], [pz, 0, -px], [-py, px, 0]])
            J = np.hstack((-skew @ v[:, None], np.eye(3)))
            J_all.append(J.T @ r)

        residuals = np.concatenate(residuals)
        J_stack = np.stack(J_all, axis=1)
        H = J_stack @ J_stack.T
        b = -J_stack @ residuals

        delta = np.linalg.solve(H, b)
        dR = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[:3])
        dt = delta[3:]

        T_update = np.eye(4)
        T_update[:3, :3] = dR
        T_update[:3, 3] = dt
        T = T_update @ T

    return T


def load_gt_poses_with_indices(file_path):
    gt_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            values = np.fromstring(line.strip(), sep=" ")
            if len(values) != 13:
                continue
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
        if fid not in gt_dict:
            continue
        rel_gt = T0_gt_inv @ gt_dict[fid]
        rel_est = T0_est_inv @ est_pose
        errors.append(np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3]))
    return np.sqrt(np.mean(np.square(errors)))


def get_available_frame_ids(base_dir):
    return sorted(
        [int(f.replace(".bin", "")) for f in os.listdir(base_dir) if f.endswith(".bin")]
    )


if __name__ == "__main__":
    base_dir = "D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0004_sync/velodyne_points/data"
    gt_pose_path = "D:/kitti360/data_poses/2013_05_28_drive_0004_sync/poses.txt"
    save_path = "trajectory_p2l.txt"

    frame_ids = get_available_frame_ids(base_dir)
    gt_dict = load_gt_poses_with_indices(gt_pose_path)
    start_idx = max(frame_ids[0], min(gt_dict.keys()))
    end_idx = start_idx + 200
    frame_gap = 5

    valid_ids = [
        fid
        for fid in range(start_idx, end_idx, frame_gap)
        if fid in gt_dict and os.path.exists(f"{base_dir}/{fid:010d}.bin")
    ]

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

        T_icp = icp_point_to_line_3d(curr_pcd, prev_pcd)
        pose = pose @ T_icp
        curr_pcd.transform(pose)
        global_map += curr_pcd
        trajectory_est.append((curr_id, pose.copy()))

        prev_pcd = curr_pcd
        prev_id = curr_id

    rmse = compute_ate_relative(trajectory_est, gt_dict)
    print(f"[EVAL] ATE RMSE (P2L): {rmse:.4f} m")

    with open(save_path, "w") as f:
        for fid, T in trajectory_est:
            t = T[:3, 3]
            quat = R.from_matrix(T[:3, :3]).as_quat()
            f.write(
                f"{fid} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n"
            )

    o3d.visualization.draw_geometries([global_map])

import open3d as o3d
import numpy as np
import argparse
import os
import glob
from scipy.spatial import cKDTree
import copy
import time

from choice_option.p_to_p_custom import run_p2p_icp
from choice_option.p_to_pl_module import run_p2pl_icp
from choice_option.gicp_module import run_gicp
from choice_option.point_to_line_icp_module import run_point_to_line_icp_custom

# ------------------ 공통 유틸 함수 ------------------
def load_point_cloud(file_path, voxel_size=0.1):
    pts = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


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


def run_icp(method, optimizer, source, target, init_trans):
    if method == "p_to_p":
        return run_p2p_icp(source, target, init_trans)
    elif method == "p_to_l":
        return run_point_to_line_icp_custom(source, target, init_trans, optimizer)
    elif method == "p_to_pl":
        return run_p2pl_icp(source, target, init_trans, optimizer)
    elif method == "gicp":
        return run_gicp(source, target, init_trans, optimizer)
    else:
        raise NotImplementedError(f"Unknown ICP method: {method}")


def get_available_frame_ids(base_dir):
    return sorted(
        [int(f.replace(".bin", "")) for f in os.listdir(base_dir) if f.endswith(".bin")]
    )


def save_frames_and_exit(args):
    os.makedirs(args.save_frames_dir, exist_ok=True)
    frame_ids = get_available_frame_ids(args.data_dir)
    gt_dict = load_gt_poses_with_indices(args.pose_path)
    start_idx = max(frame_ids[0], min(gt_dict.keys()))
    end_idx = start_idx + 200
    valid_ids = [fid for fid in range(start_idx, end_idx, args.frame_gap)
                 if fid in gt_dict and os.path.exists(f"{args.data_dir}/{fid:010d}.bin")]
    prev_id = valid_ids[0]
    prev_pcd = load_point_cloud(f"{args.data_dir}/{prev_id:010d}.bin")
    for curr_id in valid_ids[1:]:
        curr_pcd = load_point_cloud(f"{args.data_dir}/{curr_id:010d}.bin")
        T_init = np.linalg.inv(gt_dict[prev_id]) @ gt_dict[curr_id]
        T_icp, fitness, rmse = run_icp(args.method, args.optimizer, curr_pcd, prev_pcd, T_init)
        curr_pcd.transform(T_icp)
        pts = np.asarray(curr_pcd.points)
        pts_trans = (T_icp[:3,:3] @ pts.T).T + T_icp[:3,3]
        dists, _ = cKDTree(np.asarray(prev_pcd.points)).query(pts_trans)
        thr = np.mean(dists) + 0.5 * np.std(dists)
        inlier_pts = pts_trans[dists < thr]
        inlier_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_pts))
        file_path = os.path.join(args.save_frames_dir, f"frame_{curr_id:06d}.ply")
        o3d.io.write_point_cloud(file_path, inlier_pcd)
        prev_pcd = curr_pcd
        prev_id = curr_id
    print(f"Saved all frames to {args.save_frames_dir}")


def replay_saved_frames(frame_dir):
    files = sorted(glob.glob(os.path.join(frame_dir, "*.ply")))
    if not files:
        print(f"No frames found in {frame_dir}")
        return
    idx = [0]
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Frame Replay")
    def update():
        vis.clear_geometries()
        pcd = o3d.io.read_point_cloud(files[idx[0]])
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
    def next_cb(vis):
        idx[0] = (idx[0] + 1) % len(files)
        update()
        return False
    def prev_cb(vis):
        idx[0] = (idx[0] - 1) % len(files)
        update()
        return False
    vis.register_key_callback(262, next_cb)  # →
    vis.register_key_callback(263, prev_cb) # ←
    update()
    vis.run()
    vis.destroy_window()


def main(args):
    # 저장/재생 모드
    if args.save_frames_dir:
        save_frames_and_exit(args)
        return
    if args.play_frames_dir:
        replay_saved_frames(args.play_frames_dir)
        return

    # 인터랙티브 WebRTC 뷰어 모드 (나중에 실행)
    o3d.visualization.webrtc_server.enable_webrtc()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="WebRTC ICP Frame Viewer")

    frame_ids = get_available_frame_ids(args.data_dir)
    gt_dict = load_gt_poses_with_indices(args.pose_path)
    start_idx = max(frame_ids[0], min(gt_dict.keys()))
    end_idx = start_idx + 200
    valid_ids = [fid for fid in range(start_idx, end_idx, args.frame_gap)
                 if fid in gt_dict and os.path.exists(f"{args.data_dir}/{fid:010d}.bin")]
    prev_id = valid_ids[0]
    prev_pcd = load_point_cloud(f"{args.data_dir}/{prev_id:010d}.bin")

    for curr_id in valid_ids[1:]:
        curr_pcd = load_point_cloud(f"{args.data_dir}/{curr_id:010d}.bin")
        T_init = np.linalg.inv(gt_dict[prev_id]) @ gt_dict[curr_id]
        T_icp, fitness, rmse = run_icp(args.method, args.optimizer, curr_pcd, prev_pcd, T_init)
        curr_pcd.transform(T_icp)
        pts = np.asarray(curr_pcd.points)
        pts_trans = (T_icp[:3,:3] @ pts.T).T + T_icp[:3,3]
        dists, _ = cKDTree(np.asarray(prev_pcd.points)).query(pts_trans)
        thr = np.mean(dists) + 0.5 * np.std(dists)
        inlier_pts = pts_trans[dists < thr]
        inlier_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_pts))
        inlier_pcd.paint_uniform_color([0.7,0.7,0.7])

        vis.clear_geometries()
        vis.add_geometry(inlier_pcd, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        input("다음 프레임... Enter")

        prev_pcd = curr_pcd
        prev_id = curr_id

    vis.destroy_window()
    print("프레임 뷰어 종료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0003_sync/velodyne_points/data",
        help="KITTI360 bin file directory",
    )
    parser.add_argument(
        "--pose_path",
        type=str,
        default="D:/kitti360/data_poses/2013_05_28_drive_0003_sync/poses.txt",
        help="GT pose file path",
    )
    parser.add_argument("--method", choices=["p_to_p","p_to_l","p_to_pl","gicp"], default="p_to_p")
    parser.add_argument("--optimizer", choices=["svd","least_squares","gauss_newton","lm"], default="svd")
    parser.add_argument("--frame_gap", type=int, default=5)
    parser.add_argument("--save_frames_dir", type=str, default=None,
                        help="Directory to save per-frame PLYs and exit")
    parser.add_argument("--play_frames_dir", type=str, default=None,
                        help="Directory of saved PLYs to replay interactively")
    args = parser.parse_args()
    main(args)


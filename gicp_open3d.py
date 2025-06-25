import open3d as o3d
import numpy as np
import os


def load_bin_as_pcd(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def load_poses_kitti(gt_path):
    poses = []
    with open(gt_path, "r") as f:
        for i, line in enumerate(f):
            values = np.fromstring(line.strip(), sep=" ")
            if len(values) == 13:
                values = values[1:]  # 프레임 인덱스 제거
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


def run_open3d_gicp_pipeline(bin_dir, gt_pose_path):
    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    gt_poses = load_poses_kitti(gt_pose_path)
    pred_poses = [np.eye(4)]

    target = load_bin_as_pcd(os.path.join(bin_dir, bin_files[0]))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    for i in range(1, len(gt_poses)):
        print(f"[INFO] Aligning frame {i-1} → {i}")
        source = load_bin_as_pcd(os.path.join(bin_dir, bin_files[i]))
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
        )

        result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distance=1.0,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=30
            ),
        )

        T_i = pred_poses[-1] @ result.transformation
        pred_poses.append(T_i)

        target = source

    rmse = compute_ate_rmse(pred_poses[: len(gt_poses)], gt_poses)
    print(f"[EVAL] ATE RMSE: {rmse:.4f} meters")

    return pred_poses


if __name__ == "__main__":
    bin_dir = "D:/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0004_sync/velodyne_points/data"
    gt_pose_path = "D:/kitti360/data_poses/2013_05_28_drive_0004_sync/poses.txt"
    run_open3d_gicp_pipeline(bin_dir, gt_pose_path)

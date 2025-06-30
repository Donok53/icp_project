import open3d as o3d
import numpy as np
import argparse
import os
from scipy.spatial import cKDTree
import copy
import time
import csv


from scipy.spatial import KDTree
from choice_option.p_to_p_custom import run_p2p_icp
from choice_option.p_to_pl_module import run_p2pl_icp
from choice_option.gicp_module import run_gicp
from choice_option.point_to_line_icp_module import run_point_to_line_icp_custom



# ------------------ 공통 유틸 함수 ------------------
def load_point_cloud(file_path, voxel_size=0.1):
    # 1) Voxel downsample
    pts = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd = pcd.voxel_down_sample(voxel_size)
    # 2) Statistical Outlier Removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def multiscale_icp(
    method, optimizer, source, target, init_trans, scales=[0.5, 0.2, 0.1], tol=1e-6
):
    """
    coarse->fine 멀티스케일 ICP 수행
    scales: voxel_size 리스트
    """
    T_total = init_trans.copy()
    for voxel in scales:
        # 1) 다운샘플
        src_ds = source.voxel_down_sample(voxel)
        tgt_ds = target.voxel_down_sample(voxel)
        # 2) normal 재계산 (필요 시)
        src_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
        )
        tgt_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
        )
        # 3) ICP
        T_delta, fitness, rmse = run_icp(method, optimizer, src_ds, tgt_ds, np.eye(4))
        # 4) 누적 변환
        T_total = T_delta @ T_total
        # 5) source, target 에 적용
        source = copy.deepcopy(source).transform(T_total)
        target = copy.deepcopy(target)
    return T_total, fitness, rmse


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


# ------------------ 정합 알고리즘 선택 ------------------
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


# ------------------ 메인 실행 ------------------
def main(args):

    method = args.method
    optimizer = args.optimizer

    base_dir = args.data_dir
    gt_pose_path = args.pose_path

    frame_ids = get_available_frame_ids(base_dir)
    gt_dict = load_gt_poses_with_indices(gt_pose_path)

    start_idx = max(frame_ids[0], min(gt_dict.keys()))
    end_idx = start_idx + 200
    frame_gap = args.frame_gap

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

    total_icp_time = 0.0
    
    # 프레임별 metrics 기록용 리스트
    metrics = []
    
    out_aligned = os.path.join("results", "aligned_frames")
    os.makedirs(out_aligned, exist_ok=True)

    for curr_id in valid_ids[1:]:
        print(f"[INFO] Aligning frame {curr_id}")
        curr_pcd = load_point_cloud(f"{base_dir}/{curr_id:010d}.bin")

        # GT 기반 초기 추정
        T_gt_prev = gt_dict[prev_id]
        T_gt_curr = gt_dict[curr_id]
        T_init = np.linalg.inv(T_gt_prev) @ T_gt_curr

        icp_start = time.time()
        # ICP 수행
        if args.multiscale:
            T_icp, fitness, rmse = multiscale_icp(
                method, optimizer,
                curr_pcd, prev_pcd,
                T_init,
                scales=[0.5, 0.2, 0.1],
            )
        else:
            T_icp, fitness, rmse = run_icp(
                method, optimizer,
                curr_pcd, prev_pcd, T_init
            )
        icp_elapsed = time.time() - icp_start
        total_icp_time += icp_elapsed
        print(f"[TIME] Frame {curr_id} ICP time: {icp_elapsed:.3f} sec")
        print(f"[ICP] Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")

        # ── 여기서 per-frame alignment 결과를 PLY로 저장 ──
        # ── GT+ICP 결과를 하나로 합쳐서 컬러 구분된 PLY 저장 ──
        gt_aligned  = copy.deepcopy(curr_pcd)
        gt_aligned.transform(T_init)
        gt_aligned.paint_uniform_color([1, 0, 0])   # 빨강: GT

        icp_aligned = copy.deepcopy(curr_pcd)
        icp_aligned.transform(T_init @ T_icp)
        icp_aligned.paint_uniform_color([0, 0, 1])  # 파랑: ICP

        combined = gt_aligned + icp_aligned
        # method, optimizer 이름을 파일명에 추가
        combined_path = os.path.join(
            out_aligned,
            f"{method}_{optimizer}_combined_{prev_id:06d}_{curr_id:06d}.ply"
        )
        o3d.io.write_point_cloud(combined_path, combined)
        print(f"[SAVE COMBINED] {combined_path}")
        

        metrics.append((
        os.path.basename(combined_path),
        fitness,
        rmse
    ))
        
        
        # ────────────────────────────────────────────────

        # ── inlier 점만 골라서 global_map에 추가 ──
        src_pts = np.asarray(curr_pcd.points)
        src_trans = (T_icp[:3, :3] @ src_pts.T).T + T_icp[:3, 3]

        dists, _ = cKDTree(np.asarray(prev_pcd.points)).query(src_trans)
        thr = np.mean(dists) + 0.5 * np.std(dists)
        inlier_pts = src_trans[dists < thr]

        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inlier_pts)
        inlier_pcd.remove_duplicated_points()
        inlier_pcd.remove_non_finite_points()
        inlier_pcd.paint_uniform_color([0.7, 0.7, 0.7])

        global_map += inlier_pcd
        global_map = global_map.voxel_down_sample(voxel_size=0.15)
        global_map, _ = global_map.remove_statistical_outlier(
            nb_neighbors=10, std_ratio=3.0
        )

        # 궤적 업데이트 & 다음 프레임 준비
        pose = pose @ T_icp
        trajectory_est.append((curr_id, pose.copy()))
        prev_pcd = curr_pcd
        prev_id = curr_id
        
    csv_path = os.path.join(out_aligned, f"{method}_{optimizer}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","fitness","rmse"])
        w.writerows(metrics)
    print(f"[SAVE METRICS] {csv_path}")

    print("[INFO] ATE 평가 시작...")
    ate_rmse = compute_ate_relative(trajectory_est, gt_dict)
    print(f"[EVAL] ATE RMSE: {ate_rmse:.4f} meters")
    print(f"[TIME] Total ICP execution time: {total_icp_time:.3f} sec")


    # ICP gt 결과 파일

    # 기존 프로젝트나 GT 파일
    gt_file = "./choice_option/gt/2013_05_28_drive_0003_sync_gt_icp_result.ply"
    # 파일 존재 여부 체크
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"[ERROR] {gt_file} 파일을 찾을 수 없습니다.")

    # 기존 GT 포인트 클라우드 로드
    gt_pcd = o3d.io.read_point_cloud(gt_file)
    gt_pcd.paint_uniform_color([1, 0, 0])  # 빨간색으로 GT 표시

    # 현재 ICP global_map 색상 설정
    icp_pcd = global_map
    icp_pcd.paint_uniform_color([0, 0, 1])  # 파란색으로 ICP 결과 표시

    # ▶ 화면에 한 번만 띄우고
    o3d.visualization.draw_geometries(
        [gt_pcd, icp_pcd], window_name="ICP vs GT Real-Time Comparison"
    )

    # ▶ 결과 저장
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    pcd_fname = f"{method}_{optimizer}_global_map.ply"
    img_fname = f"{method}_{optimizer}_map_view.png"
    pcd_path = os.path.join(out_dir, pcd_fname)
    img_path = os.path.join(out_dir, img_fname)

    o3d.io.write_point_cloud(pcd_path, global_map)
    print(f"Saved point cloud to: {pcd_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(global_map)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(img_path)
    vis.destroy_window()
    print(f"Saved screenshot to: {img_path}")


# ------------------ Entry Point ------------------
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
    parser.add_argument(
        "--method",
        type=str,
        default="p_to_p",
        choices=["p_to_p", "p_to_l", "p_to_pl", "gicp"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="svd",
        choices=["svd", "least_squares", "gauss_newton", "lm"],
    )
    parser.add_argument("--frame_gap", type=int, default=5)
    parser.add_argument(
        "--multiscale", action="store_true", help="Enable multiscale ICP (coarse→fine)"
    )
    args = parser.parse_args()

    main(args)

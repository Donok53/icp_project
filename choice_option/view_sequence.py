# choice_option/step_viewer.py
import glob
import os
import csv
import open3d as o3d
from open3d.visualization import VisualizerWithKeyCallback

class StepViewer:
    def __init__(self, ply_dir, prefix="p_to_p_lm"):
        pattern = os.path.join(ply_dir, f"{prefix}_combined_*.ply")
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise RuntimeError(f"No {prefix}_combined_*.ply in {ply_dir}")

        csv_path = os.path.join(ply_dir, f"{prefix}_metrics.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"{csv_path} not found in {ply_dir}")
        self.metrics = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.metrics[row["filename"]] = (
                    float(row["fitness"]), float(row["rmse"])
                )

        # 3) 초기 인덱스 & 포인트클라우드 로드
        self.idx = 0
        self.pcd = o3d.io.read_point_cloud(self.files[self.idx])

    def next_frame(self, vis):
        # 다음 인덱스로 순환
        self.idx = (self.idx + 1) % len(self.files)
        new = o3d.io.read_point_cloud(self.files[self.idx])
        self.pcd.points = new.points
        if new.has_colors():
            self.pcd.colors = new.colors
        vis.update_geometry(self.pcd)

        # 터미널에 RMSE 출력
        fname = os.path.basename(self.files[self.idx])
        fitness, rmse = self.metrics.get(fname, (None, None))

        # None 체크 후 문자열 준비
        fitness_str = f"{fitness:.4f}" if fitness is not None else "N/A"
        rmse_str    = f"{rmse:.4f}"    if rmse    is not None else "N/A"

        print(f"Frame {self.idx+1}/{len(self.files)}: {fname}")
        print(f"  → Fitness: {fitness_str}, ATE RMSE: {rmse_str} m")
        return False

    def prev_frame(self, vis):
        # 이전 인덱스로 순환
        self.idx = (self.idx - 1) % len(self.files)
        new = o3d.io.read_point_cloud(self.files[self.idx])
        self.pcd.points = new.points
        if new.has_colors():
            self.pcd.colors = new.colors
        vis.update_geometry(self.pcd)

        # 터미널에 RMSE 출력
        fname = os.path.basename(self.files[self.idx])
        fitness, rmse = self.metrics.get(fname, (None, None))
        print(f"Frame {self.idx+1}/{len(self.files)}: {fname}")
        print(f"  → Fitness: {fitness:.4f}, ATE RMSE: {rmse:.4f} m")
        return False

    def run(self):
        vis = VisualizerWithKeyCallback()
        vis.create_window("Step Viewer (N: next, P: prev)")
        vis.add_geometry(self.pcd)

        # 최초 프레임 정보 출력
        fname0 = os.path.basename(self.files[0])
        fitness0, rmse0 = self.metrics.get(fname0, (None, None))
        fitness_str = f"{fitness0:.4f}" if fitness0 is not None else "N/A"
        rmse_str    = f"{rmse0:.4f}"    if rmse0    is not None else "N/A"
        print(f"Frame 1/{len(self.files)}: {fname0}")
        print(f"  → Fitness: {fitness_str}, ATE RMSE: {rmse_str} m")

        # 키 콜백 등록
        vis.register_key_callback(ord("N"), self.next_frame)
        vis.register_key_callback(ord("P"), self.prev_frame)

        print("Press 'N' for next frame, 'P' for previous frame.")
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", "-d",
        default="results/aligned_frames",
        help="combined_*.ply 및 metrics.csv 파일이 있는 디렉토리"
    )
    parser.add_argument("--prefix", "-p", default="p_to_p_svd", help="PLY/metrics 파일 prefix")
    args = parser.parse_args()
    StepViewer(args.dir).run()



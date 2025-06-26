# ICP Registration Framework

본 프로젝트는 다양한 ICP 정합 기법(Point-to-Point, Point-to-Plane, Point-to-Line, Generalized ICP)을 직접 구현하고, 다양한 최적화 방식(Least Squares, Gauss-Newton, Levenberg-Marquardt)을 적용하여 성능을 비교하는 실험적 프레임워크입니다.

---

## 🛠️ 설치 방법

### 🔗 1. 필요 패키지 설치

다음 명령어로 필요한 Python 패키지를 설치하세요:

```bash
pip install open3d numpy scipy
```

### ✅ Python 버전

* Python 3.8 이상 권장 (Open3D 호환성 기준)

---

## ⚙️ 실행 방법

```bash
python icp_framework.py \
  --data_dir path/to/KITTI360/bin_files \
  --pose_path path/to/KITTI360/poses.txt \
  --method [p_to_p | p_to_pl | p_to_l | gicp] \
  --optimizer [least_squares | gauss_newton | lm] \
  --frame_gap [int]
```

예시:

```bash
python icp_framework.py \
  --data_dir ./data/bin \
  --pose_path ./data/poses.txt \
  --method gicp \
  --optimizer lm \
  --frame_gap 5
```

---

## 📁 폴더 구성

* `icp_framework.py`: 전체 파이프라인 및 ATE 평가
* `choice_option/`: ICP 알고리즘별 모듈

  * `p_to_p_custom.py`
  * `p_to_pl_module.py`
  * `point_to_line_icp_module.py`
  * `gicp_module.py`

---

## 📌 지원 ICP 방식

| Method    | 설명                          | 지원 Optimizer                          |
| --------- | --------------------------- | ------------------------------------- |
| `p_to_p`  | Point-to-Point ICP (SVD 기반) | 없음 (고정 SVD 방식)                        |
| `p_to_pl` | Point-to-Plane ICP          | `least_squares`, `gauss_newton`, `lm` |
| `p_to_l`  | Point-to-Line ICP           | `least_squares`, `gauss_newton`, `lm` |
| `gicp`    | Generalized ICP             | `least_squares`, `gauss_newton`, `lm` |

---

## 🧠 ICP 성능 향상을 위한 공통 전략

ICP 정합 성능을 높이기 위해 아래 전략을 공통적으로 적용하는 것을 권장합니다:

1. **Voxel Downsampling 최적화**
   → `voxel_size=0.15~0.3` 범위에서 실험 필요

2. **초기 정렬 추정 개선 (T\_init)**
   → `GT`, `IMU`, `constant motion` 등 다양한 초기값 적용 실험

3. **Inlier Threshold 조정**
   → 현재 고정값(1.0\~2.0)을 adaptive하게 조정하거나, 거리 기반 가중치 적용

4. **Normal / Line direction 추정 개선**
   → 고유값 기반 필터링, 노이즈 제거, RANSAC 기반 추정 등 적용 가능

5. **Outlier 제거 사전 필터링**
   → `Statistical Outlier Removal`, `Radius Outlier Removal` 사용

6. **멀티스케일 ICP 적용 (coarse → fine)**
   → 점점 voxel 크기를 줄이며 정밀 정합 수행

7. **Robust Loss 적용**
   → Huber, Tukey 등의 `robust kernel`을 잔차 계산에 적용

---

## 📊 성능 평가

* **ATE RMSE (Absolute Trajectory Error)**
  → 정합된 trajectory와 Ground Truth 간 상대적 위치 오차를 평가 지표로 사용

---

## ✍️ TODO

*

---

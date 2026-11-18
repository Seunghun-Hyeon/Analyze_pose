import json
import os
import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R

class PoseComparatorSciPy:
    """
    SciPy의 spatial.distance 및 spatial.transform(Rotation)을 이용한
    3D 관절 spatial 오차, 십자(+) Frame 3D 회전각, 유사도 점수 산출 모듈
    """
    def __init__(self, ref_file="ref_pose.json"):
        self.ref_file = ref_file
        self.ref_data = None
        self.load_reference()

    def save_reference(self, current_angles, current_coords):
        coords_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in current_coords.items()
        }
        self.ref_data = {
            "angles": current_angles,
            "coords": coords_serializable
        }
        with open(self.ref_file, "w", encoding="utf-8") as f:
            json.dump(self.ref_data, f, indent=4)
        print(f"✅ [SciPy PoseComparator] 기준 자세 저장이 완료되었습니다: {self.ref_file}")

    def load_reference(self):
        if os.path.exists(self.ref_file):
            try:
                with open(self.ref_file, "r", encoding="utf-8") as f:
                    self.ref_data = json.load(f)
                print(f"✅ [SciPy PoseComparator] '{self.ref_file}' 로드 성공.")
            except Exception as e:
                print(f"⚠️ [SciPy PoseComparator] 파일 로드 실패: {e}")
                self.ref_data = None
        else:
            self.ref_data = None

    def calculate_scipy_error(self, current_angles, current_fixed_coords):
        """
        SciPy Euclidean Distance 기반 3D 공간 관절 변위 및 유사도 점수 계산
        """
        if self.ref_data is None:
            return {}, 100.0, 0.0

        ref_coords = self.ref_data["coords"]
        ref_angles = self.ref_data["angles"]

        spatial_errors = {}
        total_dist_error = 0.0
        
        for key in ["r_shoulder", "r_elbow", "r_wrist", "head_center", "upper_spine"]:
            if key in ref_coords and key in current_fixed_coords:
                d = distance.euclidean(ref_coords[key], current_fixed_coords[key])
                spatial_errors[f"{key}_dist_m"] = d
                total_dist_error += d

        angle_diff_sum = 0.0
        for k, ref_angle in ref_angles.items():
            if k in current_angles:
                angle_diff_sum += abs(current_angles[k] - ref_angle)

        score = max(0.0, 100.0 - (total_dist_error * 50.0 + angle_diff_sum * 1.2))
        return spatial_errors, score, total_dist_error

    def compute_cross_frame_rotation(self, spine_top, spine_bottom, left_pt, right_pt):
        """
        ★ 척추 수직축과 양 어깨/골반 수평축의 십자(+) Frame 기반 3D 회전각 연산 ★
        Returns:
            pitch: 전후 숙임 (X-Z 평면)
            roll : 좌우 기울임 (Y-Z 평면)
            yaw  : Z축 제자리 회전 (몸통/척추 비틀림)
        """
        spine_vec = spine_top - spine_bottom
        spine_norm = spine_vec / (np.linalg.norm(spine_vec) + 1e-6)

        horiz_vec = right_pt - left_pt
        horiz_norm = horiz_vec / (np.linalg.norm(horiz_vec) + 1e-6)

        forward_vec = np.cross(horiz_norm, spine_norm)
        forward_norm = forward_vec / (np.linalg.norm(forward_vec) + 1e-6)

        rot_matrix = np.vstack([horiz_norm, forward_norm, spine_norm]).T

        r = R.from_matrix(rot_matrix)
        pitch, roll, yaw = r.as_euler('xyz', degrees=True)

        return pitch, roll, yaw
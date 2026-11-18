import json
import os
import numpy as np

class PoseComparator:
    """
    기준 자세(Reference Pose) 데이터를 저장/로드하고,
    실시간 관절 데이터와의 3D 오차 및 유사도 점수(0~100점)를 산출하는 모듈
    """
    def __init__(self, ref_file="ref_pose.json"):
        self.ref_file = ref_file
        self.ref_data = None
        self.load_reference()

    def save_reference(self, current_angles, current_coords):
        """현재 자세의 관절 각도 및 3D 위치를 기준 자세 파일로 저장"""
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
        print(f"✅ [PoseComparator] 기준 자세가 '{self.ref_file}' 파일로 저장되었습니다.")

    def load_reference(self):
        """저장된 기준 자세 로드"""
        if os.path.exists(self.ref_file):
            try:
                with open(self.ref_file, "r", encoding="utf-8") as f:
                    self.ref_data = json.load(f)
                print(f"✅ [PoseComparator] '{self.ref_file}' 로드 완료.")
            except Exception as e:
                print(f"⚠️ [PoseComparator] 파일 로드 실패: {e}")
                self.ref_data = None
        else:
            self.ref_data = None

    def calculate_error(self, current_angles, current_fixed_coords):
        """
        현재 각도/3D 좌표와 기준 자세 간의 오차 산출
        Returns:
            - angle_errors (dict): 각도별 차이 (deg)
            - score (float): 100점 만점 유사도 점수
        """
        if self.ref_data is None:
            return {}, 100.0

        ref_angles = self.ref_data["angles"]
        angle_errors = {}
        total_angle_diff = 0.0

        for key, ref_val in ref_angles.items():
            if key in current_angles:
                diff = abs(current_angles[key] - ref_val)
                angle_errors[key] = diff
                total_angle_diff += diff

        # 각도 오차 기반 일치도 점수 (오차 1도당 1.5점 감점 예시)
        similarity_score = max(0.0, 100.0 - total_angle_diff * 1.5)

        return angle_errors, similarity_score
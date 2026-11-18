import numpy as np

class SpineCoordinateAnalyzer:
    def __init__(self):
        self.NOSE = 0
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28

    def extract_coordinates(self, world_landmarks):
        wl = world_landmarks

        head_center = np.array([
            (wl[self.LEFT_EAR].x + wl[self.RIGHT_EAR].x) / 2.0,
            (wl[self.LEFT_EAR].y + wl[self.RIGHT_EAR].y) / 2.0,
            (wl[self.LEFT_EAR].z + wl[self.RIGHT_EAR].z) / 2.0
        ])

        shoulder_center = np.array([
            (wl[self.LEFT_SHOULDER].x + wl[self.RIGHT_SHOULDER].x) / 2.0,
            (wl[self.LEFT_SHOULDER].y + wl[self.RIGHT_SHOULDER].y) / 2.0,
            (wl[self.LEFT_SHOULDER].z + wl[self.RIGHT_SHOULDER].z) / 2.0
        ])

        hip_center_origin = np.array([
            (wl[self.LEFT_HIP].x + wl[self.RIGHT_HIP].x) / 2.0,
            (wl[self.LEFT_HIP].y + wl[self.RIGHT_HIP].y) / 2.0,
            (wl[self.LEFT_HIP].z + wl[self.RIGHT_HIP].z) / 2.0
        ])

        upper_spine = head_center * 0.50 + shoulder_center * 0.50
        middle_spine = shoulder_center * 0.50 + hip_center_origin * 0.50
        down_spine = shoulder_center * 0.25 + hip_center_origin * 0.75

        ground_y = max(wl[self.LEFT_ANKLE].y, wl[self.RIGHT_ANKLE].y)
        ground_origin = np.array([hip_center_origin[0], ground_y, hip_center_origin[2]])

        raw_coords = {
            "head_center": head_center,
            "spine": middle_spine,
            "upper_spine": upper_spine,
            "middle_spine": middle_spine,
            "down_spine": down_spine,
            "shoulder_center": shoulder_center,
            "hip_center": hip_center_origin,
            "r_shoulder": np.array([wl[self.RIGHT_SHOULDER].x, wl[self.RIGHT_SHOULDER].y, wl[self.RIGHT_SHOULDER].z]),
            "l_shoulder": np.array([wl[self.LEFT_SHOULDER].x, wl[self.LEFT_SHOULDER].y, wl[self.LEFT_SHOULDER].z]),
            "r_hip": np.array([wl[self.RIGHT_HIP].x, wl[self.RIGHT_HIP].y, wl[self.RIGHT_HIP].z]),
            "l_hip": np.array([wl[self.LEFT_HIP].x, wl[self.LEFT_HIP].y, wl[self.LEFT_HIP].z]),
            "r_elbow": np.array([wl[self.RIGHT_ELBOW].x, wl[self.RIGHT_ELBOW].y, wl[self.RIGHT_ELBOW].z]),
            "l_elbow": np.array([wl[self.LEFT_ELBOW].x, wl[self.LEFT_ELBOW].y, wl[self.LEFT_ELBOW].z]),
            "r_wrist": np.array([wl[self.RIGHT_WRIST].x, wl[self.RIGHT_WRIST].y, wl[self.RIGHT_WRIST].z]),
            "l_wrist": np.array([wl[self.LEFT_WRIST].x, wl[self.LEFT_WRIST].y, wl[self.LEFT_WRIST].z]),
            "r_knee": np.array([wl[self.RIGHT_KNEE].x, wl[self.RIGHT_KNEE].y, wl[self.RIGHT_KNEE].z]),
            "l_knee": np.array([wl[self.LEFT_KNEE].x, wl[self.LEFT_KNEE].y, wl[self.LEFT_KNEE].z]),
            "r_ankle": np.array([wl[self.RIGHT_ANKLE].x, wl[self.RIGHT_ANKLE].y, wl[self.RIGHT_ANKLE].z]),
            "l_ankle": np.array([wl[self.LEFT_ANKLE].x, wl[self.LEFT_ANKLE].y, wl[self.LEFT_ANKLE].z]),
        }

        ground_fixed_coords = {
            key: val - ground_origin for key, val in raw_coords.items()
        }

        return raw_coords, ground_fixed_coords

    def calculate_angle_3d(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
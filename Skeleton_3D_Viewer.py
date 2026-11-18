import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Skeleton3DViewer:
    """
    SpineCoordinateAnalyzer에서 추출된 ground_fixed_coords(3D 미터 좌표)를
    Matplotlib 3D 뷰포트 공간상에 전신(상반신+하반신) 스켈레톤으로 실시간 시각화
    """
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 전신 (상체 + 하체 + 머리) 3D 연결 선분 구조
        self.connections = [
            # 머리 & 척추
            ("head_center", "upper_spine"),
            ("upper_spine", "middle_spine"),
            ("middle_spine", "down_spine"),
            ("down_spine", "hip_center"),
            # 상체 및 팔
            ("r_shoulder", "l_shoulder"),
            ("r_shoulder", "r_elbow"),
            ("r_elbow", "r_wrist"),
            ("l_shoulder", "l_elbow"),
            ("l_elbow", "l_wrist"),
            # 하체 및 다리
            ("r_hip", "l_hip"),
            ("r_hip", "r_knee"),
            ("r_knee", "r_ankle"),
            ("l_hip", "l_knee"),
            ("l_knee", "l_ankle"),
            # 상하체 연결 (어깨 중심 - 골반 중심)
            ("shoulder_center", "hip_center")
        ]

    def update_3d_plot(self, ground_fixed_coords):
        self.ax.clear()

        xs, ys, zs = [], [], []
        for key, pos in ground_fixed_coords.items():
            xs.append(pos[0])       # X: 좌우
            ys.append(pos[2])       # Z: 앞뒤 (Depth)
            zs.append(-pos[1])      # Y: 높이 (Height)

        self.ax.scatter(xs, ys, zs, c='red', s=35, marker='o')

        for p1_key, p2_key in self.connections:
            if p1_key in ground_fixed_coords and p2_key in ground_fixed_coords:
                p1 = ground_fixed_coords[p1_key]
                p2 = ground_fixed_coords[p2_key]
                self.ax.plot(
                    [p1[0], p2[0]], 
                    [p1[2], p2[2]], 
                    [-p1[1], -p2[1]], 
                    c='blue', linewidth=2
                )

        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(0.0, 2.2)

        self.ax.set_xlabel('X: Left/Right (m)')
        self.ax.set_ylabel('Z: Depth (m)')
        self.ax.set_zlabel('Y: Height (m)')
        self.ax.set_title("Real-Time 3D Full-Body Skeleton")

        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)
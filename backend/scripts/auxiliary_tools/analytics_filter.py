import numpy as np
import cv2


class TacticalPhysicsFilter:
    def __init__(self):
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        self.min_players_to_judge = 6
        self.min_hull_area = 150.0
        self.max_out_of_bounds_ratio = 0.3
        self.min_std_dev = 2.0

    def is_frame_humanly_possible(self, radar_pts):
        # Filter out Nones just in case
        valid_pts = [pt for pt in radar_pts if pt is not None]
        if len(valid_pts) < self.min_players_to_judge:
            return False

        pts_array = np.array(valid_pts, dtype=np.float32)

        # RULE 1: Bounds Check
        out_of_bounds_count = 0
        for pt in pts_array:
            if (abs(pt[0]) > (self.pitch_length / 2) + 5.0 or
                    abs(pt[1]) > (self.pitch_width / 2) + 5.0):
                out_of_bounds_count += 1

        if (out_of_bounds_count / len(pts_array)) > self.max_out_of_bounds_ratio:
            return False

        # RULE 2: Single File / Variance Check
        std_x = np.std(pts_array[:, 0])
        std_y = np.std(pts_array[:, 1])
        if std_x < self.min_std_dev or std_y < self.min_std_dev:
            return False

        # RULE 3: Black Hole / Area Check
        hull_input = pts_array.reshape(-1, 1, 2)
        try:
            hull = cv2.convexHull(hull_input)
            area = cv2.contourArea(hull)
            if area < self.min_hull_area:
                return False
        except Exception:
            return False

        return True

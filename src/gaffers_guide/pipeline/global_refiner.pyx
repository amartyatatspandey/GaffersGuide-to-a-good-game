# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""Global offline trajectory refinement for chunk-level TacticalFrame timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


PlayerFactory = Callable[[int | None, str, list[float] | None], object]
FrameFactory = Callable[[int, list[object]], object]


@dataclass(slots=True)
class GlobalRefiner:
    """Refine player trajectories across a full chunk before analytics."""

    fps: float = 30.0
    outlier_speed_mps: float = 12.0
    cap_speed_mps: float = 10.0
    max_gap_frames: int = 75
    savgol_window: int = 21
    savgol_polyorder: int = 2

    def refine(
        self,
        frames: list[object],
        *,
        frame_factory: FrameFactory,
        player_factory: PlayerFactory,
    ) -> list[object]:
        """
        Return a refined list of frames with healed/interpolated/smoothed player tracks.
        """
        if not frames:
            return []

        frame_indices = [int(f.frame_idx) for f in frames]
        start_idx = min(frame_indices)
        end_idx = max(frame_indices)
        total = end_idx - start_idx + 1
        frame_pos_by_idx = {idx: idx - start_idx for idx in frame_indices}

        tracks: dict[tuple[int, str], np.ndarray] = {}
        for frame in frames:
            pos = frame_pos_by_idx[int(frame.frame_idx)]
            for player in frame.players:
                if player.id is None:
                    continue
                key = (int(player.id), str(player.team))
                if key not in tracks:
                    arr = np.full((total, 2), np.nan, dtype=np.float64)
                    tracks[key] = arr
                if player.radar_pt is not None:
                    tracks[key][pos, 0] = float(player.radar_pt[0])
                    tracks[key][pos, 1] = float(player.radar_pt[1])

        refined_tracks: dict[tuple[int, str], np.ndarray] = {}
        for key, arr in tracks.items():
            healed = arr.copy()
            self._nullify_outliers(healed)
            self._interpolate_short_gaps(healed)
            self._smooth_track(healed)
            self._cap_velocity(healed)
            self._smooth_track(healed)
            refined_tracks[key] = healed

        rebuilt: list[object] = []
        for frame in frames:
            pos = frame_pos_by_idx[int(frame.frame_idx)]
            players_out: list[object] = []
            for (player_id, team), arr in refined_tracks.items():
                x = arr[pos, 0]
                y = arr[pos, 1]
                if np.isnan(x) or np.isnan(y):
                    continue
                players_out.append(
                    player_factory(player_id, team, [float(x), float(y)])
                )
            rebuilt.append(frame_factory(int(frame.frame_idx), players_out))
        return rebuilt

    def _nullify_outliers(self, arr: np.ndarray) -> None:
        valid_idx = np.flatnonzero(~np.isnan(arr[:, 0]) & ~np.isnan(arr[:, 1]))
        if valid_idx.size < 2:
            return
        for i in range(1, valid_idx.size):
            prev_i = int(valid_idx[i - 1])
            cur_i = int(valid_idx[i])
            dt = (cur_i - prev_i) / self.fps
            if dt <= 0:
                continue
            dist = float(np.linalg.norm(arr[cur_i] - arr[prev_i]))
            speed = dist / dt
            if speed > self.outlier_speed_mps:
                arr[cur_i] = np.array([np.nan, np.nan], dtype=np.float64)

    def _interpolate_short_gaps(self, arr: np.ndarray) -> None:
        valid_mask = ~np.isnan(arr[:, 0]) & ~np.isnan(arr[:, 1])
        idx = 0
        n = arr.shape[0]
        while idx < n:
            if valid_mask[idx]:
                idx += 1
                continue
            gap_start = idx
            while idx < n and not valid_mask[idx]:
                idx += 1
            gap_end = idx - 1
            gap_len = gap_end - gap_start + 1
            left = gap_start - 1
            right = gap_end + 1
            if (
                left >= 0
                and right < n
                and valid_mask[left]
                and valid_mask[right]
                and gap_len <= self.max_gap_frames
            ):
                x_interp = interp1d(
                    [left, right],
                    [arr[left, 0], arr[right, 0]],
                    kind="linear",
                )
                y_interp = interp1d(
                    [left, right],
                    [arr[left, 1], arr[right, 1]],
                    kind="linear",
                )
                fill_idx = np.arange(gap_start, gap_end + 1)
                arr[fill_idx, 0] = x_interp(fill_idx)
                arr[fill_idx, 1] = y_interp(fill_idx)
                valid_mask[fill_idx] = True

    def _smooth_track(self, arr: np.ndarray) -> None:
        valid_mask = ~np.isnan(arr[:, 0]) & ~np.isnan(arr[:, 1])
        valid_idx = np.flatnonzero(valid_mask)
        if valid_idx.size < 5:
            return

        x = arr[valid_idx, 0]
        y = arr[valid_idx, 1]
        window = min(self.savgol_window, valid_idx.size if valid_idx.size % 2 == 1 else valid_idx.size - 1)
        if window < 5:
            return
        poly = min(self.savgol_polyorder, window - 1)
        if poly < 1:
            return

        arr[valid_idx, 0] = savgol_filter(x, window_length=window, polyorder=poly, mode="interp")
        arr[valid_idx, 1] = savgol_filter(y, window_length=window, polyorder=poly, mode="interp")

    def _cap_velocity(self, arr: np.ndarray) -> None:
        valid_idx = np.flatnonzero(~np.isnan(arr[:, 0]) & ~np.isnan(arr[:, 1]))
        if valid_idx.size < 2:
            return

        original = arr.copy()
        for i in range(1, valid_idx.size):
            prev_i = int(valid_idx[i - 1])
            cur_i = int(valid_idx[i])
            dt = (cur_i - prev_i) / self.fps
            if dt <= 0:
                continue
            max_step = self.cap_speed_mps * dt
            delta = arr[cur_i] - arr[prev_i]
            dist = float(np.linalg.norm(delta))
            if dist <= max_step or dist == 0.0:
                continue

            clipped = arr[prev_i] + (delta / dist) * max_step
            arr[cur_i] = clipped

            # Blend back toward original over the next few known points.
            blend_count = 8
            for k in range(1, blend_count + 1):
                j_pos = i + k
                if j_pos >= valid_idx.size:
                    break
                j = int(valid_idx[j_pos])
                alpha = k / (blend_count + 1)
                arr[j] = (1.0 - alpha) * arr[j] + alpha * original[j]

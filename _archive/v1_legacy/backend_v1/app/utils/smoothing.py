"""
Player velocity smoothing for tracking data (Model 2 – Fatigue Index).

Logic ported from references/LaurieOnTracking/Metrica_Velocities.py:
- Velocity from position difference over timestep; optional Savitzky–Golay or
  moving-average smoothing applied per half to avoid smoothing across half-time.
- Supports Pandas DataFrame (Metrica-style columns) or list of dicts.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

FilterKind = Literal["Savitzky-Golay", "moving average"]

# Scipy is optional; Savitzky–Golay uses it; moving average uses numpy only.
try:
    from scipy.signal import savgol_filter

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _ensure_odd_window(window: int) -> int:
    """Savitzky–Golay requires odd window length."""
    return window if window % 2 == 1 else max(3, window + 1)


def _smooth_halves(
    v: np.ndarray,
    period: np.ndarray,
    filter_kind: FilterKind,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """
    Smooth velocity array per half (first half / second half) to avoid crossing half-time.

    v, period: same length. period is 1 or 2. Returns smoothed v (same shape).
    """
    out = np.asarray(v, dtype=float).copy()
    window = _ensure_odd_window(window)

    for half in (1, 2):
        mask = period == half
        if not np.any(mask):
            continue
        v_half = out[mask]
        n = len(v_half)
        # Need window <= n and window > polyorder for savgol
        w = min(window, n) if n >= 3 else n
        if w < 2:
            continue
        if w % 2 == 0:
            w -= 1
        if w < 2:
            continue
        po = min(polyorder, w - 1)

        if filter_kind == "Savitzky-Golay" and _HAS_SCIPY:
            try:
                smoothed = savgol_filter(v_half, window_length=w, polyorder=po, mode="nearest")
                out[mask] = smoothed
            except Exception as e:
                logger.debug("savgol_filter failed for half %s: %s", half, e)
        elif filter_kind == "moving average" or not _HAS_SCIPY:
            ma = np.ones(w) / w
            out[mask] = np.convolve(v_half, ma, mode="same")

    return out


def calc_player_velocities_arrays(
    time: np.ndarray,
    period: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    smoothing: bool = True,
    filter_kind: FilterKind = "Savitzky-Golay",
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smoothed velocities for one player from time and position arrays.

    Velocity is diff(position)/diff(time); values above maxspeed are set to NaN.
    Smoothing is applied per half (period 1 vs 2) so half-time is not blended.

    Args:
        time: Time in seconds, shape (N,).
        period: Period index (1 or 2), shape (N,).
        x, y: Position in meters, shape (N,).
        smoothing: Whether to apply the filter.
        filter_kind: 'Savitzky-Golay' or 'moving average'.
        window: Smoothing window size (frames). Made odd for Savitzky–Golay.
        polyorder: Polynomial order for Savitzky–Golay (1 = linear).
        maxspeed: Max plausible speed (m/s); above this, velocity set to NaN.

    Returns:
        vx, vy, speed: Each shape (N,). First element is NaN (no diff at index 0).
    """
    time = np.asarray(time, dtype=float).ravel()
    period = np.asarray(period, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(time)
    if len(period) != n or len(x) != n or len(y) != n:
        raise ValueError("time, period, x, y must have the same length")

    dt = np.diff(time)
    dt = np.concatenate([[np.nan], dt])
    vx = np.diff(x) / np.where(dt[1:] != 0, dt[1:], np.nan)
    vx = np.concatenate([[np.nan], vx])
    vy = np.diff(y) / np.where(dt[1:] != 0, dt[1:], np.nan)
    vy = np.concatenate([[np.nan], vy])

    if maxspeed > 0:
        raw_speed = np.sqrt(vx * vx + vy * vy)
        bad = raw_speed > maxspeed
        vx = np.where(bad, np.nan, vx)
        vy = np.where(bad, np.nan, vy)

    if smoothing:
        vx = _smooth_halves(vx, period, filter_kind, window, polyorder)
        vy = _smooth_halves(vy, period, filter_kind, window, polyorder)

    speed = np.sqrt(vx * vx + vy * vy)
    return vx, vy, speed


def calc_player_velocities_records(
    records: list[dict[str, Any]],
    time_key: str,
    period_key: str,
    player_ids: list[str],
    x_suffix: str = "_x",
    y_suffix: str = "_y",
    vx_suffix: str = "_vx",
    vy_suffix: str = "_vy",
    speed_suffix: str = "_speed",
    smoothing: bool = True,
    filter_kind: FilterKind = "Savitzky-Golay",
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = 12.0,
) -> list[dict[str, Any]]:
    """
    Add smoothed velocity fields to each record for each player.

    Reads time_key, period_key, and for each player_id the keys
    {player_id}{x_suffix}, {player_id}{y_suffix}. Writes
    {player_id}{vx_suffix}, {player_id}{vy_suffix}, {player_id}{speed_suffix}.

    Args:
        records: List of frame dicts (e.g. [{"Time [s]": 0.0, "Period": 1, "Home_1_x": 10, ...}]).
        time_key: Key for time in seconds.
        period_key: Key for period (1 or 2).
        player_ids: List of player identifiers (e.g. ["Home_1", "Away_3"]).
        x_suffix, y_suffix: Suffixes for position keys.
        vx_suffix, vy_suffix, speed_suffix: Suffixes for output velocity keys.
        smoothing, filter_kind, window, polyorder, maxspeed: Passed to core velocity routine.

    Returns:
        New list of dicts with same keys plus velocity keys (input list is not mutated).
    """
    if not records or not player_ids:
        return [dict(r) for r in records]

    time = np.array([r[time_key] for r in records], dtype=float)
    period = np.array([r[period_key] for r in records], dtype=float)

    out = [dict(r) for r in records]
    for pid in player_ids:
        xk, yk = pid + x_suffix, pid + y_suffix
        if xk not in records[0] or yk not in records[0]:
            continue
        x = np.array([r[xk] for r in records], dtype=float)
        y = np.array([r[yk] for r in records], dtype=float)
        vx, vy, speed = calc_player_velocities_arrays(
            time, period, x, y,
            smoothing=smoothing,
            filter_kind=filter_kind,
            window=window,
            polyorder=polyorder,
            maxspeed=maxspeed,
        )
        vxk, vyk, sk = pid + vx_suffix, pid + vy_suffix, pid + speed_suffix
        for i, rec in enumerate(out):
            rec[vxk] = float(vx[i])
            rec[vyk] = float(vy[i])
            rec[sk] = float(speed[i])
    return out


def calc_player_velocities_df(
    df: Any,
    time_col: str = "Time [s]",
    period_col: str = "Period",
    player_ids: list[str] | None = None,
    x_suffix: str = "_x",
    y_suffix: str = "_y",
    smoothing: bool = True,
    filter_kind: FilterKind = "Savitzky-Golay",
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = 12.0,
) -> Any:
    """
    Add smoothed velocity columns to a tracking DataFrame (Metrica-style).

    Drops existing velocity-related columns (vx, vy, speed, ax, ay, acceleration),
    then for each player adds {player_id}_vx, _vy, _speed.

    Args:
        df: Pandas DataFrame with time_col, period_col, and {player_id}_x, _y per player.
        time_col: Column name for time in seconds.
        period_col: Column name for period (1 or 2).
        player_ids: If None, inferred from columns: those ending in x_suffix (stem = player id).
        x_suffix, y_suffix: Suffixes for position columns.
        smoothing, filter_kind, window, polyorder, maxspeed: Passed to core velocity routine.

    Returns:
        New DataFrame with velocity columns added (type same as input).
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("calc_player_velocities_df requires a pandas DataFrame")

    # Remove existing velocity/acceleration columns
    drop = [
        c for c in df.columns
        if c.split("_")[-1] in ("vx", "vy", "ax", "ay", "speed", "acceleration")
    ]
    team = df.drop(columns=drop, errors="ignore")

    if player_ids is None:
        candidate = [c for c in team.columns if c.endswith(x_suffix)]
        player_ids = sorted(set(c[:-len(x_suffix)] for c in candidate if (c[:-len(x_suffix)] + y_suffix) in team.columns))

    if not player_ids:
        return team

    time = team[time_col].values
    period = team[period_col].values

    for pid in player_ids:
        xcol, ycol = pid + x_suffix, pid + y_suffix
        if xcol not in team.columns or ycol not in team.columns:
            continue
        x = team[xcol].values
        y = team[ycol].values
        vx, vy, speed = calc_player_velocities_arrays(
            time, period, x, y,
            smoothing=smoothing,
            filter_kind=filter_kind,
            window=window,
            polyorder=polyorder,
            maxspeed=maxspeed,
        )
        team[pid + "_vx"] = vx
        team[pid + "_vy"] = vy
        team[pid + "_speed"] = speed

    return team


def calc_player_velocities(
    data: Any,
    *,
    time_col: str = "Time [s]",
    period_col: str = "Period",
    time_key: str | None = None,
    period_key: str | None = None,
    player_ids: list[str] | None = None,
    smoothing: bool = True,
    filter_kind: FilterKind = "Savitzky-Golay",
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = 12.0,
) -> Any:
    """
    Add smoothed player velocities to tracking data (DataFrame or list of dicts).

    Dispatches to calc_player_velocities_df or calc_player_velocities_records.
    For list of dicts, time_key/period_key default to time_col/period_col.

    Returns:
        DataFrame or list of dicts with _vx, _vy, _speed per player.
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and isinstance(data, pd.DataFrame):
        return calc_player_velocities_df(
            data,
            time_col=time_col,
            period_col=period_col,
            player_ids=player_ids,
            smoothing=smoothing,
            filter_kind=filter_kind,
            window=window,
            polyorder=polyorder,
            maxspeed=maxspeed,
        )
    if isinstance(data, list) and data and isinstance(data[0], dict):
        tkey = time_key if time_key is not None else time_col
        pkey = period_key if period_key is not None else period_col
        return calc_player_velocities_records(
            data,
            time_key=tkey,
            period_key=pkey,
            player_ids=player_ids or _player_ids_from_records(data[0]),
            smoothing=smoothing,
            filter_kind=filter_kind,
            window=window,
            polyorder=polyorder,
            maxspeed=maxspeed,
        )
    raise TypeError("data must be a pandas DataFrame or a non-empty list of dicts")


def _player_ids_from_records(record: dict[str, Any], x_suffix: str = "_x", y_suffix: str = "_y") -> list[str]:
    """Infer player ids from first record: keys ending with x_suffix with matching y_suffix."""
    ids = []
    for k in record:
        if k.endswith(x_suffix):
            stem = k[: -len(x_suffix)]
            if (stem + y_suffix) in record:
                ids.append(stem)
    return sorted(set(ids))

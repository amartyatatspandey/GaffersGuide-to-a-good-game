"""
Pitch control surface: probability that a team gains possession if the ball is moved to a location.

Logic extracted from LaurieOnTracking/Metrica_PitchControl.py (Spearman 2018).
Math only; no plotting. Uses numpy for vector calculations.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

# Default field (meters). Metrica convention: length x width.
DEFAULT_FIELD_LENGTH = 106.0
DEFAULT_FIELD_WIDTH = 68.0


class PitchControlParams(TypedDict, total=False):
    """Model parameters for pitch control (Spearman 2018). Keys match default_model_params()."""

    max_player_speed: float
    reaction_time: float
    tti_sigma: float
    kappa_def: float
    lambda_att: float
    lambda_def: float
    lambda_gk: float
    average_ball_speed: float
    int_dt: float
    max_int_time: float
    model_converge_tol: float
    time_to_control_att: float
    time_to_control_def: float


def default_model_params(time_to_control_veto: float = 3.0) -> PitchControlParams:
    """
    Default parameters for the pitch control model (Spearman 2018).

    time_to_control_veto: ignore a player if P(control) < 10^(-time_to_control_veto).
    """
    params: PitchControlParams = {
        "max_player_speed": 5.0,
        "reaction_time": 0.7,
        "tti_sigma": 0.45,
        "kappa_def": 1.0,
        "lambda_att": 4.3,
        "average_ball_speed": 15.0,
    }
    params["lambda_def"] = 4.3 * params["kappa_def"]
    params["lambda_gk"] = params["lambda_def"] * 3.0
    params["int_dt"] = 0.04
    params["max_int_time"] = 10.0
    params["model_converge_tol"] = 0.01
    factor = time_to_control_veto * np.log(10) * (
        np.sqrt(3) * params["tti_sigma"] / np.pi + 1.0 / params["lambda_att"]
    )
    params["time_to_control_att"] = factor
    params["time_to_control_def"] = time_to_control_veto * np.log(10) * (
        np.sqrt(3) * params["tti_sigma"] / np.pi + 1.0 / params["lambda_def"]
    )
    return params


def time_to_intercept(
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    vmax: float,
    reaction_time: float,
) -> float:
    """
    Time for a player to reach target: react, then run at vmax from reacted position.

    position, target: (2,) in meters. velocity: (2,) m/s.
    """
    position = np.asarray(position, dtype=float).reshape(2)
    velocity = np.asarray(velocity, dtype=float).reshape(2)
    target = np.asarray(target, dtype=float).reshape(2)
    r_reaction = position + velocity * reaction_time
    return reaction_time + float(np.linalg.norm(target - r_reaction)) / vmax


def probability_intercept_ball(T: float, time_to_intercept_val: float, tti_sigma: float) -> float:
    """
    P(player arrives at target by time T) from Spearman 2018 sigmoid.

    T: time (s). time_to_intercept_val: expected arrival time (s). tti_sigma: spread (s).
    """
    x = (T - time_to_intercept_val) * (np.pi / np.sqrt(3.0)) / tti_sigma
    return float(1.0 / (1.0 + np.exp(-x)))


def _lambda_for_player(is_goalkeeper: bool, params: PitchControlParams) -> float:
    return params["lambda_gk"] if is_goalkeeper else params["lambda_def"]


def calculate_pitch_control_at_target(
    target_position: np.ndarray,
    attacking_players: list[tuple[np.ndarray, np.ndarray]],
    defending_players: list[tuple[np.ndarray, np.ndarray, bool]],
    ball_start_pos: np.ndarray | None,
    params: PitchControlParams,
) -> tuple[float, float]:
    """
    Pitch control probability at a single target (attacking, defending).

    attacking_players: list of (position (2,), velocity (2,)).
    defending_players: list of (position (2,), velocity (2,), is_goalkeeper).
    ball_start_pos: (2,) or None (ball already at target).
    Returns (PPCF_att, PPCF_def); should sum to ~1 within model_converge_tol.
    """
    target = np.asarray(target_position, dtype=float).reshape(2)
    vmax = params["max_player_speed"]
    rt = params["reaction_time"]
    tti_sigma = params["tti_sigma"]
    ball_speed = params["average_ball_speed"]

    if ball_start_pos is None or np.any(np.isnan(ball_start_pos)):
        ball_travel_time = 0.0
    else:
        ball_start_pos = np.asarray(ball_start_pos, dtype=float).reshape(2)
        ball_travel_time = float(np.linalg.norm(target - ball_start_pos)) / ball_speed

    def tti_att(p: tuple[np.ndarray, np.ndarray]) -> float:
        return time_to_intercept(p[0], p[1], target, vmax, rt)

    def tti_def(p: tuple[np.ndarray, np.ndarray, bool]) -> float:
        return time_to_intercept(p[0], p[1], target, vmax, rt)

    tau_min_att = min(tti_att(p) for p in attacking_players) if attacking_players else np.inf
    tau_min_def = min(tti_def(p) for p in defending_players) if defending_players else np.inf

    ttc_att = params["time_to_control_att"]
    ttc_def = params["time_to_control_def"]

    if tau_min_att - max(ball_travel_time, tau_min_def) >= ttc_def:
        return 0.0, 1.0
    if tau_min_def - max(ball_travel_time, tau_min_att) >= ttc_att:
        return 1.0, 0.0

    # Filter to players close enough in time to matter
    att = [p for p in attacking_players if tti_att(p) - tau_min_att < ttc_att]
    def_ = [p for p in defending_players if tti_def(p) - tau_min_def < ttc_def]

    int_dt = params["int_dt"]
    max_int_time = params["max_int_time"]
    tol = params["model_converge_tol"]

    dT_array = np.arange(
        ball_travel_time - int_dt,
        ball_travel_time + max_int_time,
        int_dt,
        dtype=float,
    )
    PPCFatt = np.zeros_like(dT_array)
    PPCFdef = np.zeros_like(dT_array)

    # Per-player cumulative PPCF (we integrate dPPCF/dT * dt)
    pcf_att = [0.0] * len(att)
    pcf_def = [0.0] * len(def_)
    tti_att_vals = [tti_att(p) for p in att]
    tti_def_vals = [tti_def(p) for p in def_]
    lambda_def_vals = [_lambda_for_player(p[2], params) for p in def_]

    i = 1
    while i < len(dT_array):
        T = dT_array[i]
        rem = 1.0 - PPCFatt[i - 1] - PPCFdef[i - 1]
        for j, p in enumerate(att):
            dppcf = rem * probability_intercept_ball(T, tti_att_vals[j], tti_sigma) * params["lambda_att"]
            pcf_att[j] += dppcf * int_dt
        for j, p in enumerate(def_):
            dppcf = rem * probability_intercept_ball(T, tti_def_vals[j], tti_sigma) * lambda_def_vals[j]
            pcf_def[j] += dppcf * int_dt
        PPCFatt[i] = sum(pcf_att)
        PPCFdef[i] = sum(pcf_def)
        ptot = PPCFatt[i] + PPCFdef[i]
        if ptot >= 1.0 - tol:
            return float(PPCFatt[i]), float(PPCFdef[i])
        i += 1

    return float(PPCFatt[i - 1]), float(PPCFdef[i - 1])


def pitch_control_grid(
    players: list[tuple[np.ndarray, np.ndarray, Literal["att", "def"], bool]],
    ball_position: np.ndarray,
    field_length: float = DEFAULT_FIELD_LENGTH,
    field_width: float = DEFAULT_FIELD_WIDTH,
    n_cells_x: int = 50,
    params: PitchControlParams | None = None,
) -> np.ndarray:
    """
    Compute 2D pitch control grid (attacking team probability) over the field.

    players: list of (position (2,), velocity (2,), team_side 'att'|'def', is_goalkeeper).
    ball_position: (2,) in meters (same coordinate system as positions).
    field_length, field_width: in meters.
    n_cells_x: number of cells along length; n_cells_y derived from aspect ratio.

    Returns 2D array of shape (n_cells_y, n_cells_x), values in [0, 1] (attacking control).
    """
    if params is None:
        params = default_model_params()

    attacking = [(np.asarray(p[0]), np.asarray(p[1])) for p in players if p[2] == "att"]
    defending = [(np.asarray(p[0]), np.asarray(p[1]), p[3]) for p in players if p[2] == "def"]

    ball = np.asarray(ball_position, dtype=float).reshape(2)

    n_cells_y = int(round(n_cells_x * field_width / field_length))
    dx = field_length / n_cells_x
    dy = field_width / n_cells_y
    xgrid = (np.arange(n_cells_x) + 0.5) * dx - field_length / 2.0
    ygrid = (np.arange(n_cells_y) + 0.5) * dy - field_width / 2.0

    grid = np.zeros((n_cells_y, n_cells_x), dtype=float)
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            target = np.array([xgrid[j], ygrid[i]])
            p_att, _ = calculate_pitch_control_at_target(
                target, attacking, defending, ball, params
            )
            grid[i, j] = p_att
    return grid

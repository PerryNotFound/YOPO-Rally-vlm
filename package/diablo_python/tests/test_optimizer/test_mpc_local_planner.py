import time
from typing import Callable

import numpy as np
import pytest

from diablo_python.optimizer import MPCLocalPlanner
from diablo_python.utils import Draw_MPC_tracking


@pytest.fixture
def planner():
    planner = MPCLocalPlanner()
    planner.setup_opti_params(
        v_max=2,
        omega_max=2 * np.pi,
        acc_max=5,
        alpha_max=2 * np.pi,
        sampling_time=0.1,
        horizon=10,
    )
    planner.bake_optim_func(method="sqpmethod")
    return planner


@pytest.fixture
def init_opti_variables(planner: MPCLocalPlanner):
    init_state = np.array([0, 0, 0])
    current_state = init_state.copy()

    u = np.zeros((planner.N, 2))
    state_pred = np.zeros((planner.N + 1, 3))
    state_ref = np.zeros((planner.N + 1, 3))
    return init_state, current_state, u, state_pred, state_ref


def visualize(
    init_state: np.ndarray,
    sim_period: float,
    state_pred_list: np.ndarray,
    state_ref_list: np.ndarray,
    state_real_list: np.ndarray,
):
    frame_delta_time = 66
    step = int(frame_delta_time / sim_period / 1000)

    return Draw_MPC_tracking(
        robot_states=state_real_list[::step],
        init_state=init_state,
        reference_states=state_ref_list[::step],
        prediction_states=state_pred_list[::step],
        export_fig=False,
        interval=frame_delta_time,
    )


def get_reference_traj(
    t: float,
    N: int,
    Ts: float,
    pattern: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
):
    pass
    t_pred = np.linspace(t, t + (N + 1) * Ts, N + 1)
    x_ref, y_ref = pattern(t_pred)
    if np.all(x_ref == x_ref[0]) and np.all(y_ref == y_ref[0]):
        theta_ref = np.ones_like(x_ref) * np.arctan2(y_ref[0], x_ref[0])
    else:
        theta_ref = np.arctan2(np.diff(y_ref), np.diff(x_ref))
        theta_ref = np.append(theta_ref, theta_ref[-1])
    state_ref = np.column_stack((x_ref, y_ref, theta_ref))
    return state_ref


def circle_pattern(t_pred: np.ndarray):
    radius = 8 / np.pi
    omega_circle = np.pi / 8
    CENTER = np.array([1, 1])

    x_ref = CENTER[0] + radius * np.cos(omega_circle * t_pred)
    y_ref = CENTER[1] + radius * np.sin(omega_circle * t_pred)
    return x_ref, y_ref


def point_pattern(t_pred: np.ndarray):
    x_ref = np.ones_like(t_pred) * 4
    y_ref = np.ones_like(t_pred) * 4
    return x_ref, y_ref


def line_pattern(t_pred: np.ndarray):
    vel = 0.5
    x_ref = vel * t_pred
    y_ref = vel * t_pred
    return x_ref, y_ref


def sine_pattern(t_pred: np.ndarray):
    vel = 1
    sine_T = 8
    x_ref = vel * t_pred
    y_ref = vel * np.sin(np.pi * 2 * t_pred / sine_T)
    return x_ref, y_ref


def polynomial_pattern(t_pred: np.ndarray):
    vel = 0.5
    x_ref = vel * t_pred / 4
    y_ref = vel * (t_pred / 4) ** 2
    return x_ref, y_ref


def loveknot_pattern(t_pred: np.ndarray):
    vel = 2
    T = 12
    x_ref = vel * np.sin(2 * np.pi * t_pred / T)
    y_ref = vel / 2 * np.sin(4 * np.pi * t_pred / T)
    return x_ref, y_ref


# @pytest.mark.skip(reason="This test is too slow to run on CI")
@pytest.mark.parametrize(
    "traj_pattern",
    [
        circle_pattern,
        line_pattern,
        point_pattern,
        sine_pattern,
        polynomial_pattern,
        loveknot_pattern,
    ],
)
def test_mpc_local_planner(
    planner: MPCLocalPlanner,
    traj_pattern: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    init_opti_variables: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
):
    should_visualize = False
    init_state, current_state, u, state_pred, state_ref = init_opti_variables

    sim_time_total = 15
    sim_period = 0.002
    mpc_period = 0.05

    sim_time_steps = int(sim_time_total / sim_period) + 1
    state_pred_list = np.zeros((sim_time_steps, planner.N + 1, 3))
    state_ref_list = np.zeros((sim_time_steps, planner.N + 1, 3))
    state_real_list = np.zeros((sim_time_steps, 3))
    # start simulation
    start_cal_time = time.time()
    mpc_last_time = 0
    mpc_counter = 0
    mpc_cal_time = 0
    for idx in range(sim_time_steps):
        sim_time = idx * sim_period
        if sim_time - mpc_last_time >= mpc_period:
            start_mpc_cal_time = time.time()
            mpc_last_time = sim_time
            # add noise to the current_state
            noise = np.random.normal(0, 0.1, current_state.shape)
            current_state_measured = current_state + noise
            # update reference trajectory
            state_ref = get_reference_traj(
                sim_time, planner.N, planner.Ts, traj_pattern
            )
            # solve optimization problem
            u, state_pred = planner.solve_optim_warm_start(
                x_ref=state_ref, x_cur=current_state_measured
            )
            end_mpc_cal_time = time.time()
            mpc_cal_time += end_mpc_cal_time - start_mpc_cal_time
            mpc_counter += 1

        current_state = MPCLocalPlanner.shift_movement(current_state, u[0], sim_period)
        state_ref_list[idx] = state_ref
        state_pred_list[idx] = state_pred
        state_real_list[idx] = current_state

    end_cal_time = time.time()
    print(f"Total time: {end_cal_time - start_cal_time:.3f}s")
    print(
        f"Average time: {(end_cal_time - start_cal_time) / sim_time_steps*1000:.3f}ms"
    )
    print(
        f"Average time per MPC: {mpc_cal_time / mpc_counter*1000:.3f}ms, {mpc_counter} MPCs"
    )
    if should_visualize:
        visualize(
            init_state, sim_period, state_pred_list, state_ref_list, state_real_list
        )

    start_eval_time = 10
    tolerance = 0.3

    start_compare_idx = int(start_eval_time / sim_period)
    result_pos = np.allclose(
        state_ref_list[start_compare_idx:, 0, :2],
        state_real_list[start_compare_idx:, :2],
        atol=tolerance,
    )

    assert result_pos, "Position tracking failed, error too large!"

import matplotlib.pyplot as plt
import numpy as np
import pytest

from diablo_python.geotiff import GeoTiffCostLut, GeoTiffReader
from diablo_python.optimizer import PolyTrajOptimizer
from diablo_python.utils import (
    generate_splines,
    show_cost,
    visualize_traj,
    visualize_velocity,
)


@pytest.fixture
def optimizer(lut_bspline: GeoTiffCostLut):
    optimizer = PolyTrajOptimizer()
    optimizer.setup_optim_params(
        v_max=2,
        t_end=10,
        look_ahead_distance=7.5,
        cost_lut=lut_bspline,
        sampling_number=30,
    )
    optimizer.bake_optim_func()
    return optimizer


@pytest.fixture
def init_opti_variables():
    pos_start = np.array([20, -12])
    vel_start = np.array([1, 0])
    goal = pos_start + np.array([10, 0])
    return (
        pos_start,
        vel_start,
        goal,
    )


@pytest.fixture
def init_opti_variables_random():
    pos_start = np.random.uniform(-1, 1, 2) * 40 + np.array([50, -50])
    # theta_center is randomly generated in the range of [-pi, pi]
    theta_center = np.random.rand() * 2 * np.pi - np.pi

    # vel_start's direction is randomly generated in the range of [-pi, pi]
    angle_offset = np.random.uniform(-np.pi / 4, np.pi / 4) + theta_center
    vel_start = np.random.uniform(1, 2) * np.array(
        [np.cos(angle_offset), np.sin(angle_offset)]
    )

    # goal is randomly generated
    radius = np.random.uniform(5, 10)
    angle_offset = np.random.uniform(-np.pi / 4, np.pi / 4) + theta_center
    goal = pos_start + np.array(
        [radius * np.cos(angle_offset), radius * np.sin(angle_offset)]
    )
    return pos_start, vel_start, theta_center, goal


@pytest.mark.skip(reason="Show cost image will block the test")
def test_show_cost(lut: GeoTiffCostLut, geotiff_reader: GeoTiffReader):
    show_cost(lut, geotiff_reader)
    plt.show()


# @pytest.mark.skip(reason="Need to fix the test")
def test_poly_traj_optimizer_candidate(
    optimizer: PolyTrajOptimizer,
    init_opti_variables: tuple[np.ndarray, np.ndarray, np.ndarray],
    geotiff_reader: GeoTiffReader,
    lut_linear: GeoTiffCostLut,
    candidate_num: int,
):
    visualize = False
    pos_start, vel_start, goal = init_opti_variables

    (
        pos_end_candidate,
        vel_end_candidate,
        cost_candidate,
    ) = optimizer.solve_optim_candidate(
        pos_start=pos_start,
        vel_start=vel_start,
        goal=goal,
        candidate_num=candidate_num,
    )
    pos_end_init_guess_candidate = optimizer.p_end_init_guess_candidate
    print(
        f"pos_end_candidate: {pos_end_candidate}, \nvel_end_candidate \ncost_candidate: {cost_candidate}"
    )
    if visualize:
        t_eval = np.linspace(0, optimizer.t_end, 100)
        min_cost_idx: int = np.argmin(cost_candidate)

        plt.figure()
        plt.gca().set_aspect("equal", adjustable="box")
        show_cost(lut_linear, geotiff_reader)

        pos_spline_x_list, vel_spline_x_list, pos_spline_y_list, vel_spline_y_list = (
            generate_splines(
                candidate_num,
                pos_start,
                vel_start,
                pos_end_candidate,
                vel_end_candidate,
                t_eval,
            )
        )

        visualize_traj(
            candidate_num=candidate_num,
            pos_start=pos_start,
            goal=goal,
            pos_end_init_guess_candidate=pos_end_init_guess_candidate,
            t_eval=t_eval,
            min_cost_idx=min_cost_idx,
            pos_spline_x_list=pos_spline_x_list,
            pos_spline_y_list=pos_spline_y_list,
        )

        visualize_velocity(
            vel_start,
            t_eval,
            min_cost_idx,
            candidate_num,
            vel_spline_x_list,
            vel_spline_y_list,
        )
        plt.show()


test_num = 10


@pytest.mark.parametrize("test", [i for i in range(test_num)])
# @pytest.mark.skip(reason="Need to fix the test")
def test_poly_traj_optimizer_candidate_random(
    optimizer: PolyTrajOptimizer,
    init_opti_variables_random: tuple[np.ndarray, np.ndarray, float, np.ndarray],
    geotiff_reader: GeoTiffReader,
    lut_linear: GeoTiffCostLut,
    test: int,
    candidate_num: int,
):
    visualize = False
    pos_start, vel_start, theta_center, goal = init_opti_variables_random

    try:
        (
            pos_end_candidate,
            vel_end_candidate,
            cost_candidate,
        ) = optimizer.solve_optim_candidate(
            pos_start=pos_start,
            vel_start=vel_start,
            goal=goal,
            candidate_num=candidate_num,
            candidate_theta_center=theta_center,
        )
    except Exception as e:
        raise AssertionError(f"Error: {e}") from e

    pos_end_init_guess_candidate = optimizer.p_end_init_guess_candidate
    min_cost_idx: int = np.argmin(cost_candidate)
    print(f"cost_candidate: {cost_candidate}, min_cost: {cost_candidate[min_cost_idx]}")
    if visualize:
        if test == 0:
            plt.figure(figsize=(10, 10))
            plt.gca().set_aspect("equal", adjustable="box")
            show_cost(lut_linear, geotiff_reader)
        t_eval = np.linspace(0, optimizer.t_end, 100)

        (
            pos_spline_x_list,
            vel_spline_x_list,
            pos_spline_y_list,
            vel_spline_y_list,
        ) = generate_splines(
            candidate_num,
            pos_start,
            vel_start,
            pos_end_candidate,
            vel_end_candidate,
            t_eval,
        )
        visualize_traj(
            candidate_num=candidate_num,
            pos_start=pos_start,
            goal=goal,
            pos_end_init_guess_candidate=pos_end_init_guess_candidate,
            t_eval=t_eval,
            min_cost_idx=min_cost_idx,
            pos_spline_x_list=pos_spline_x_list,
            pos_spline_y_list=pos_spline_y_list,
        )

        if test == test_num - 1:
            plt.show()

    assert cost_candidate[min_cost_idx] < 50, "Cost too high"

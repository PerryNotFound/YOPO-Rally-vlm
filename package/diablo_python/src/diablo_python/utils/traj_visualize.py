import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.interpolate import CubicHermiteSpline

from diablo_python.geotiff import GeoTiffCostLut, GeoTiffReader


# TODO: draw trajectory(with cost) and velocity in 2D at the first row, velocity in the second row
# TODO: Use fastplotlib to speed up the visualization, or with matplotlib's ion mode
def visualize_traj(
    candidate_num: int,
    pos_start: np.ndarray,
    goal: np.ndarray,
    pos_end_init_guess_candidate: np.ndarray,
    t_eval: np.ndarray,
    min_cost_idx: int,
    pos_spline_x_list: list[CubicHermiteSpline],
    pos_spline_y_list: list[CubicHermiteSpline],
):
    # TODO: visualize theta limit
    for i in range(candidate_num):
        pos_end_init_guess = pos_end_init_guess_candidate[i]
        # visualize
        p_eval_x = pos_spline_x_list[i](t_eval)
        p_eval_y = pos_spline_y_list[i](t_eval)
        if i != min_cost_idx:
            plt.plot(p_eval_x, p_eval_y, label=f"traj_{i}", alpha=0.5, linestyle="--")
        else:
            plt.plot(p_eval_x, p_eval_y, label=f"traj_{i}")
        plt.plot(pos_end_init_guess[0], pos_end_init_guess[1], "ro")

    plt.plot(pos_start[0], pos_start[1], "bo")
    plt.plot(goal[0], goal[1], "go")
    # plt.legend()
    plt.title("Trajectory")


def visualize_velocity(
    vel_start: np.ndarray,
    t_eval: np.ndarray,
    min_cost_idx: int,
    candidate_num: int,
    vel_spline_x_list: list[CubicHermiteSpline],
    vel_spline_y_list: list[CubicHermiteSpline],
):
    plt.figure(figsize=(10, 8))

    v_max_abs = 0
    for i in range(candidate_num):
        v_eval = np.array([vel_spline_x_list[i](t_eval), vel_spline_y_list[i](t_eval)])
        v_eval_abs_max = np.max(np.abs(v_eval))
        v_max_abs = v_eval_abs_max if v_eval_abs_max > v_max_abs else v_max_abs

        if i != min_cost_idx:
            plt.plot(v_eval[0], v_eval[1], label=f"vel_{i}", alpha=0.5, linestyle="--")
        else:
            plt.plot(v_eval[0], v_eval[1], label=f"vel_{i}")
        plt.plot(vel_start[0], vel_start[1], "ro")

    plt.legend()
    v_max_abs *= 1.1
    plt.axhline(0, color="black", linewidth=1.5)
    plt.axvline(0, color="black", linewidth=1.5)
    plt.xlim(-v_max_abs, v_max_abs)
    plt.ylim(-v_max_abs, v_max_abs)
    plt.grid(True)
    plt.title("Velocity in 2D")
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()


def generate_splines(
    candidate_num, pos_start, vel_start, pos_end_candidate, vel_end_candidate, t_eval
):
    pos_spline_x_list: list[CubicHermiteSpline] = []
    vel_spline_x_list: list[CubicHermiteSpline] = []
    pos_spline_y_list: list[CubicHermiteSpline] = []
    vel_spline_y_list: list[CubicHermiteSpline] = []
    for i in range(candidate_num):
        pos_spline_x = CubicHermiteSpline(
            [0, t_eval[-1]],
            [pos_start[0], pos_end_candidate[i, 0]],
            [vel_start[0], vel_end_candidate[i, 0]],
        )
        pos_spline_x_list.append(pos_spline_x)

        pos_spline_y = CubicHermiteSpline(
            [0, t_eval[-1]],
            [pos_start[1], pos_end_candidate[i, 1]],
            [vel_start[1], vel_end_candidate[i, 1]],
        )
        pos_spline_y_list.append(pos_spline_y)

        vel_spline_x = pos_spline_x.derivative()
        vel_spline_x_list.append(vel_spline_x)

        vel_spline_y = pos_spline_y.derivative()
        vel_spline_y_list.append(vel_spline_y)
    return pos_spline_x_list, vel_spline_x_list, pos_spline_y_list, vel_spline_y_list


def show_cost(lut: GeoTiffCostLut, geotiff_reader: GeoTiffReader):
    cost = lut.cost
    bounds = geotiff_reader._dataset.bounds
    plt.imshow(
        cost,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        norm=colors.LogNorm(),
    )
    # limit the colorbar range
    plt.colorbar()

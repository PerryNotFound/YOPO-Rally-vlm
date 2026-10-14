import matplotlib.pyplot as plt
import numpy as np

from diablo_python.geotiff import GeoTiffCostLut, GeoTiffReader
from diablo_python.utils import (
    generate_splines,
    show_cost,
    visualize_traj,
)

from .cloud_compare import CloudCompare
from .data_file_finder import DataFileFinder
from .traj_opti import TrajOpti


class DataGenerator:
    def __init__(self, data_file_finder: DataFileFinder):
        self._data_file_finder = data_file_finder

    def setup_cloud_compare(self, cloud_compare: CloudCompare):
        self._cloud_compare = cloud_compare

    def setup_traj_opt(self, traj_opt: TrajOpti):
        self._traj_opt = traj_opt
        self._traj_opt.set_toml_opt_pattern(self._data_file_finder.toml_opt_pattern)

    def process(self, show_progress: bool = True):
        terrain_files, tree_files, toml_files, _ = (
            self._data_file_finder.get_all_files()
        )
        self._cloud_compare.process(terrain_files, tree_files)
        self._traj_opt.process(toml_files, show_progress)

    def visualize_toml_opt_data(
        self,
        tiff_file_prefix: str,
    ):
        _, _, _, toml_opt_files = self._data_file_finder.get_all_files()
        tiff_files = self._data_file_finder.get_all_tiff_files(tiff_file_prefix)

        T_END = 10
        LOOK_AHEAD_DISTANCE = 7.5
        candidate_theta_range = np.pi / 2
        for toml_opt, tiff in zip(toml_opt_files, tiff_files, strict=True):
            plt.figure(figsize=(10, 10))
            (
                _,
                yaw_start_list,
                pos_start_list,
                vel_start_list,
                goal_list,
                pos_end_list,
                vel_end_list,
                cost_list,
            ) = self._data_file_finder.process_toml_file(toml_opt)
            plt.gca().set_aspect("equal", adjustable="box")
            geotiff_reader = GeoTiffReader(tiff, 2, 3)
            lut_linear = GeoTiffCostLut(geotiff_reader)
            lut_linear.bake_lut("linear")
            show_cost(lut_linear, geotiff_reader)
            t_eval = np.linspace(0, T_END, 100)

            for i in range(len(pos_start_list)):
                yaw_start = yaw_start_list[i]
                pos_start = pos_start_list[i]
                vel_start = vel_start_list[i]
                goal = goal_list[i]
                pos_end_candidate = pos_end_list[i]
                vel_end_candidate = vel_end_list[i]
                cost = cost_list[i]
                candidate_num = len(vel_end_candidate)
                min_cost_idx = np.argmin(cost)

                pos_end_init_guess_candidate = np.zeros((candidate_num, 2))
                for j in range(candidate_num):
                    theta = (
                        j - candidate_num / 2 + 0.5
                    ) * candidate_theta_range / candidate_num + yaw_start
                    theta_vec = np.array([np.cos(theta), np.sin(theta)])
                    pos_end_init_guess_candidate[j] = (
                        pos_start + LOOK_AHEAD_DISTANCE * theta_vec
                    )

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

        plt.show()
        pass

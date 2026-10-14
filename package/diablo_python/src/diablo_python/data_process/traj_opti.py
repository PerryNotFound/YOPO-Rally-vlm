import glob
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import rtoml
from rich.progress import Progress

from diablo_python.geotiff import GeoTiffCostLut, GeoTiffReader
from diablo_python.optimizer import PolyTrajOptimizer


class TrajOpti:
    random_vel_max = 1.5
    random_vel_min = 0.5
    random_radius_max = 10
    random_radius_min = 5

    def __init__(
        self,
        tiff_file_prefix: str,
        random_num_per_point: int = 10,
        candidate_num: int = 6,
        fixed_random_seed: Optional[int] = None,
        multi_process: bool = False,
    ):
        self._random_num_per_point = random_num_per_point
        self._candidate_num = candidate_num
        self._tiff_file_prefix = tiff_file_prefix
        self._multi_process = multi_process

        self._fixed_random_seed = fixed_random_seed
        self.progress: Optional[Progress] = None

    def set_toml_opt_pattern(self, toml_opt_pattern: str):
        self._toml_opt_pattern = toml_opt_pattern

    def process(self, toml_data_files: list[str], show_progress: bool = True):
        toml_data_num = len(toml_data_files)

        # TODO: Use Pool in _optimize_per_pos_start
        if self._multi_process:
            # TODO: Memory usage is high, need to optimize
            with ProcessPoolExecutor() as exeutor:
                results = exeutor.map(
                    self._optimize_per_scene,
                    toml_data_files,
                    range(toml_data_num),
                )
                for _ in results:
                    pass
        else:
            self.progress = Progress(disable=not show_progress)
            self.progress.start()
            self._overall_task_id = self.progress.add_task(
                "Overall", total=toml_data_num
            )
            for i in range(toml_data_num):
                self._optimize_per_scene(toml_data_files[i], i)
            self.progress.stop()

    def _optimize_per_scene(self, toml_data_file: str, i: int):
        if self._fixed_random_seed is not None:
            np.random.seed(self._fixed_random_seed + i)
        if self._multi_process:
            if i == 0:
                # TODO: There should be a better way to handle progress bar
                self.progress = Progress()
                self.progress.start()
                self._overall_task_id = self.progress.add_task("Overall", total=1)

        if self.progress:
            traj_opt_task_id = self.progress.add_task(
                f"Optimizing trajectories {i}", total=1
            )
        f = open(toml_data_file, "r", encoding="utf-8")
        toml_data = rtoml.load(f)
        f.close()

        camera_hfov = toml_data["depthCameraHorizontalFOV"]
        tiff_map_file = glob.glob(
            os.path.join(
                os.path.dirname(toml_data_file), f"{self._tiff_file_prefix}*.tif"
            )
        )
        assert len(tiff_map_file) == 1, "Only one tiff file is allowed"
        tiff_map_file = tiff_map_file[0]

        if self.progress:
            self.progress.console.print(f"Processing {tiff_map_file}")
        geotiff_reader = GeoTiffReader(tiff_map_file, dip_band=2, roughness_band=3)
        cost_lut = GeoTiffCostLut(geotiff_reader)
        cost_lut.bake_lut(inflation_radius=0.3)
        optimizer = PolyTrajOptimizer()
        optimizer.setup_optim_params(
            v_max=2,
            t_end=10,
            look_ahead_distance=7.5,
            cost_lut=cost_lut,
            sampling_number=30,
        )
        optimizer.bake_optim_func()

        data_array: dict = toml_data["dataArray"]

        for data in data_array:
            self._optimize_per_pos_start(data, camera_hfov, optimizer)
            advance = 1 / len(data_array)
            if self.progress:
                self.progress.update(traj_opt_task_id, advance=advance)
                self.progress.update(self._overall_task_id, advance=advance)

        file_folder = os.path.dirname(toml_data_file)
        toml_data_opt_file = os.path.join(file_folder, self._toml_opt_pattern)

        with open(toml_data_opt_file, "w", encoding="utf-8") as f:
            rtoml.dump(toml_data, f)
        if self.progress:
            self.progress.stop()

    def _optimize_per_pos_start(
        self,
        data: dict,
        camera_hfov: float,
        optimizer: PolyTrajOptimizer,
    ):
        pos_start = np.array(data["posStart"], dtype=np.float32)
        yaw_start = data["yawStart"]

        vel_start_list = []
        goal_list = []
        pos_end_list = []
        vel_end_list = []
        cost_list = []

        for _ in range(self._random_num_per_point):
            angle_offset = np.deg2rad(
                np.random.uniform(-camera_hfov / 2, camera_hfov / 2) + yaw_start
            )
            vel_start = np.random.uniform(
                self.random_vel_min, self.random_vel_max
            ) * np.array([np.cos(angle_offset), np.sin(angle_offset)])
            radius = np.random.uniform(self.random_radius_min, self.random_radius_max)
            angle_offset = np.deg2rad(
                np.random.uniform(-camera_hfov / 2, camera_hfov / 2) + yaw_start
            )
            goal = pos_start + radius * np.array(
                [np.cos(angle_offset), np.sin(angle_offset)]
            )
            try:
                (
                    pos_end_candidate,
                    vel_end_candidate,
                    cost_candidate,
                ) = optimizer.solve_optim_candidate(
                    pos_start,
                    vel_start,
                    goal,
                    candidate_num=self._candidate_num,
                    candidate_theta_center=np.deg2rad(yaw_start),
                )
            except Exception as e:
                print(f"failed: {e}")
                continue

            vel_start_list.append(vel_start.tolist())
            goal_list.append(goal.tolist())
            cost_list.append(cost_candidate.tolist())
            pos_end_list.append(pos_end_candidate.tolist())
            vel_end_list.append(vel_end_candidate.tolist())

            min_cost_idx: int = np.argmin(cost_candidate)

            if cost_candidate[min_cost_idx] > 50:
                print("Warning: cost too high")

        data["velStart"] = vel_start_list
        data["goal"] = goal_list
        data["posEnd"] = pos_end_list
        data["velEnd"] = vel_end_list

        data["cost"] = cost_list
        return data

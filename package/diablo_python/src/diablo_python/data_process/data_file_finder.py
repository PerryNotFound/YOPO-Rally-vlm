import glob
import os
import re

import numpy as np
import rtoml


class DataFileFinder:
    scene_prefix = "Scene_"
    scene_pattern = re.compile(rf"{scene_prefix}\d+")
    terrain_pattern = "terrain.ply"
    tree_pattern = "tree.ply"
    toml_pattern = "data.toml"
    toml_opt_pattern = "data_opt.toml"

    def __init__(self, data_folder_path: str):
        self._data_folder_path = data_folder_path

    def get_scene_folders(self):
        return [
            f
            for f in glob.glob(
                os.path.join(self._data_folder_path, self.scene_prefix + "*")
            )
            if os.path.isdir(f) and self.scene_pattern.search(f)
        ]

    def get_all_files(self):
        folders = self.get_scene_folders()
        patterns = [
            self.terrain_pattern,
            self.tree_pattern,
            self.toml_pattern,
            self.toml_opt_pattern,
        ]
        files: dict[str, list[str]] = {pattern: [] for pattern in patterns}

        for folder in folders:
            for pattern in patterns:
                files[pattern].extend(glob.glob(os.path.join(folder, pattern)))

        return (
            files[self.terrain_pattern],
            files[self.tree_pattern],
            files[self.toml_pattern],
            files[self.toml_opt_pattern],
        )

    def get_all_tiff_files(self, tiff_file_prefix):
        tiff_files: list[str] = []
        folders = self.get_scene_folders()
        tiff_pattern = tiff_file_prefix + "*.tif"
        for folder in folders:
            tiff_files.extend(glob.glob(os.path.join(folder, tiff_pattern)))
        return tiff_files

    @staticmethod
    def process_toml_file(toml_file):
        with open(toml_file, "r", encoding="utf-8") as f:
            toml_data = rtoml.load(f)
        image_paths: list[str] = []
        yaw_starts: list[np.ndarray] = []
        pos_starts: list[np.ndarray] = []
        vel_starts: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        pos_ends: list[np.ndarray] = []
        vel_ends: list[np.ndarray] = []
        costs: list[np.ndarray] = []
        for data in toml_data["dataArray"]:
            if "imageFileName" in data:
                imageFileNameData = data["imageFileName"]
            elif "imageFileNameList" in data and len(data["imageFileNameList"]) == 1:
                imageFileNameData = data["imageFileNameList"][0]
            else:
                raise ValueError(
                    "imageFileNameList should be a string or a list with one string element"
                )
            image_path = os.path.join(
                os.path.dirname(toml_file), "Textures", imageFileNameData
            )

            yaw_start = np.asarray(data["yawStart"], dtype=np.float32)
            yaw_start = np.deg2rad(yaw_start)
            pos_start = np.asarray(data["posStart"], dtype=np.float32)
            vel_start = np.asarray(data["velStart"], dtype=np.float32)
            goal = np.asarray(data["goal"], dtype=np.float32)
            pos_end = np.asarray(data["posEnd"], dtype=np.float32)
            vel_end = np.asarray(data["velEnd"], dtype=np.float32)
            cost = np.asarray(data["cost"], dtype=np.float32)

            random_num = len(vel_start)
            assert (
                random_num == len(goal) == len(vel_end) == len(pos_end) == len(cost)
            ), (
                "Lengths of random_num, goal, vel_end, pos_end, and cost should be the same"
            )
            assert vel_end.shape[1] == pos_end.shape[1] == cost.shape[1], (
                "Lengths of vel_end, pos_end, and cost should be the same"
            )
            image_paths.extend([image_path] * random_num)
            yaw_starts.extend([yaw_start] * random_num)
            pos_starts.extend([pos_start] * random_num)
            vel_starts.extend(vel_start)
            goals.extend(goal)
            pos_ends.extend(pos_end)
            vel_ends.extend(vel_end)
            costs.extend(cost)

        return (
            image_paths,
            yaw_starts,
            pos_starts,
            vel_starts,
            goals,
            pos_ends,
            vel_ends,
            costs,
        )

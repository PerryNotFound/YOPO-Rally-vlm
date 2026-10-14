import os
import time

import cv2
import numpy as np
import torch
from rich import print
from torch import Tensor
from torch.utils.data import Dataset

from diablo_python.data_process import DataFileFinder

from .yopo_transforms import YOPOTrainTransform


class YOPODataset(Dataset):
    def __init__(
        self,
        data_folder_path: str,
        transform: YOPOTrainTransform,
        image_size: tuple[int, int] = (160, 32),
        bake_cache=True,
    ):
        self._data_folder_path = data_folder_path
        self._image_size = image_size
        self._transform = transform
        self._bake_cache = bake_cache
        # TODO: Make a image_dict and state_output_list inside the cache_dict
        self._cache_dict: dict[int | str, tuple[Tensor, ...] | Tensor | list[str]] = {}

        self._load_data_files()
        if self._bake_cache:
            self._generate_cache()

    def _generate_cache(self):
        CACHE_DICT_FILE_NAME = "cache_dict.pth"
        CACHE_DICT_PATH = os.path.join(self._data_folder_path, CACHE_DICT_FILE_NAME)
        IMAGE_PATH_LIST_STR = "image_path_list"
        if os.path.exists(CACHE_DICT_PATH):
            # FIXME: Check if the cache_dict matches the current data_folder_path
            cache_dict = torch.load(CACHE_DICT_PATH, weights_only=True)
            if (
                self._data_folder_path in cache_dict
                and IMAGE_PATH_LIST_STR in cache_dict
            ):
                self._cache_dict = cache_dict
                self._image_path_list = cache_dict[IMAGE_PATH_LIST_STR]
                print("[bold green]Cache loaded successfully![/bold green]")
                return

        print("[bold yellow]Cache not found, generating cache...[/bold yellow]")
        self._cache_dict[self._data_folder_path] = tuple()
        self._cache_dict[IMAGE_PATH_LIST_STR] = self._image_path_list

        start_time = time.time()
        for i in range(len(self)):
            self[i]
        print(f"Cache generation time: {time.time() - start_time}")
        torch.save(self._cache_dict, CACHE_DICT_PATH)
        print("[bold green]Cache generated successfully![/bold green]")

    def _load_data_files(self):
        data_process = DataFileFinder(self._data_folder_path)
        (
            self._terrain_files,
            self._tree_files,
            self._toml_files,
            self._toml_opt_files,
        ) = data_process.get_all_files()
        self._image_path_list: list[str] = []
        self._yaw_start_list: list[np.ndarray] = []
        self._pos_start_list: list[np.ndarray] = []
        self._vel_start_list: list[np.ndarray] = []
        self._goal_list: list[np.ndarray] = []
        self._pos_end_list: list[np.ndarray] = []
        self._vel_end_list: list[np.ndarray] = []
        self._cost_list: list[np.ndarray] = []

        for toml_file in self._toml_opt_files:
            (
                image_paths,
                yaw_starts,
                pos_starts,
                vel_starts,
                goals,
                pos_ends,
                vel_ends,
                costs,
            ) = DataFileFinder.process_toml_file(toml_file)
            self._image_path_list.extend(image_paths)
            self._yaw_start_list.extend(yaw_starts)
            self._pos_start_list.extend(pos_starts)
            self._vel_start_list.extend(vel_starts)
            self._goal_list.extend(goals)
            self._pos_end_list.extend(pos_ends)
            self._vel_end_list.extend(vel_ends)
            self._cost_list.extend(costs)

    def __len__(self):
        return len(self._image_path_list)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image_path = self._image_path_list[index]
        if self._bake_cache:
            if index in self._cache_dict:
                state, output = self._cache_dict[index]
            else:
                state, output = self._prepare_state_output(index)
                self._cache_dict[index] = state, output
            if image_path in self._cache_dict:
                image = self._cache_dict[image_path]
            else:
                image = self._prepare_image(image_path)
                self._cache_dict[image_path] = image

        else:
            image = self._prepare_image(image_path)
            state, output = self._prepare_state_output(index)
        return image, state, output

    def _prepare_image(self, image_path: str):
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[
            :, :, 0
        ]

        # Convert depth image to distance image
        _height, width = image.shape
        fov_rad = np.pi / 2
        half_width = width / 2
        x_indices = np.arange(width) - half_width
        angles = np.arctan(x_indices / half_width * np.tan(fov_rad / 2))
        image = image / np.cos(angles).astype(np.float32)

        image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, axis=-1)

        image_tensor = torch.from_numpy(image)

        # (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor

    def _prepare_state_output(self, index: int):
        yaw_start = self._yaw_start_list[index]
        yaw_start = np.expand_dims(yaw_start, axis=0)

        yaw_start_tensor = torch.from_numpy(yaw_start)
        vel_start_tensor = torch.from_numpy(self._vel_start_list[index])
        pos_start_tensor = torch.from_numpy(self._pos_start_list[index])
        goal_tensor = torch.from_numpy(self._goal_list[index])
        pos_end_tensor = torch.from_numpy(self._pos_end_list[index].T)
        vel_end_tensor = torch.from_numpy(self._vel_end_list[index].T)
        cost_tensor = torch.from_numpy(self._cost_list[index].T)

        state_tensor, output_tensor = self._transform(
            yaw_start_tensor,
            pos_start_tensor,
            vel_start_tensor,
            goal_tensor,
            pos_end_tensor,
            vel_end_tensor,
            cost_tensor,
        )

        return state_tensor, output_tensor

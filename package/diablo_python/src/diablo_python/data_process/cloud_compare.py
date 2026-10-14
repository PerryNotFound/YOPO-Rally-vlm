# TODO: Postpone imports
import glob
import os
import shutil
import subprocess
from functools import reduce

from rich import print
from rich.status import Status


class CloudCompare:
    def __init__(self, tiff_file_prefix: str) -> None:
        self._setup_env()
        cc_exec_paths = [
            "CloudCompare",
            "flatpak run org.cloudcompare.CloudCompare",
        ]

        for path in cc_exec_paths:
            if shutil.which(path.split()[0]):
                print(f"[bold green]Using CloudCompare executable: {path}[/]")
                self._cc_exec_path = path
                break
        else:
            raise FileNotFoundError(
                f"CloudCompare executable not found or cannot be executed. "
                f"Tried: {', '.join(cc_exec_paths)}"
            )
        self._tiff_file_prefix = tiff_file_prefix

    def _setup_env(self):
        # Remove redundant environment variables to prevent issues with CloudCompare
        self._env = os.environ.copy()
        self._env.pop("VIRTUAL_ENV", None)
        self._env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        self._env.pop("QT_QPA_FONTDIR", None)

    def _generate_config_commands(self):
        # fmt: off
        config_commands = self._cc_exec_path.split() + [
            "-VERBOSITY", "3",
            "-SILENT",
            "-AUTO_SAVE", "OFF",
        ]
        return config_commands
        # fmt: on

    def _append_process_commands(self, terrain_file: str, tree_file: str):
        tiff_file_path = os.path.dirname(terrain_file)
        if glob.glob(os.path.join(tiff_file_path, f"{self._tiff_file_prefix}*")):
            print(f"[bold yellow]Skipping {terrain_file} and {tree_file}[/]")
            return []

        # fmt: off
        commands = [
            "-O", terrain_file,
            "-O", tree_file,
            "-MERGE_CLOUDS",
            "-REMOVE_ALL_SFS",
            "-OCTREE_NORMALS", "0.3", "-ORIENT", "PLUS_Z", "-MODEL", "LS",
            "-NORMALS_TO_DIP",
            "-REMOVE_SF", "1",
            "-ROUGH", "0.3",
            "-NO_TIMESTAMP",
            "-RENAME_ENTITIES", self._tiff_file_prefix,
            # "-SAVE_CLOUDS",
            "-RASTERIZE", "-GRID_STEP", "0.3", "-OUTPUT_RASTER_Z_AND_SF",
            "-CLEAR"
        ]
        # fmt: on
        return commands

    def process(self, terrain_files: list[str], tree_files: list[str]):
        process_cmd = reduce(
            lambda acc, files: acc + self._append_process_commands(*files),
            zip(terrain_files, tree_files, strict=True),
            self._generate_config_commands(),
        )
        with Status("Processing terrain and tree files..."):
            subprocess.run(process_cmd, env=self._env, check=True)
        pass

import os

import casadi as ca
import cv2
import numpy as np
from rich import print

from .geotiff_reader import GeoTiffReader


class GeoTiffCostLut:
    def __init__(self, geotiff_reader: GeoTiffReader):
        self._geotiff_reader = geotiff_reader

    @staticmethod
    def cost_callable(
        dip_data: np.ndarray,
        roughness_data: np.ndarray,
        grid_size: float,
        inflation_pixel_radius: int,
    ):
        max_dip = 30
        max_roughness = 0.1

        # -------- FIXED PARAMETERS --------
        distance_threshold = 1
        distance_scale_factor = 1
        # -------- FIXED PARAMETERS --------

        dip_overflow = dip_data > max_dip
        roughness_overflow = roughness_data > max_roughness
        total_overflow = dip_overflow | roughness_overflow

        # Dilate the overflow area by some pixels
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inflation_pixel_radius * 2 + 1, inflation_pixel_radius * 2 + 1),
        )
        total_overflow = cv2.dilate(total_overflow.astype(np.uint8), kernel)

        # calculate the distance transform of the overflow
        collision_map_inverted = total_overflow.astype(np.uint8) * 255
        collision_map = cv2.bitwise_not(collision_map_inverted)
        udf_foreground = cv2.distanceTransform(
            collision_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        udf_background = cv2.distanceTransform(
            collision_map_inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        sdf = (udf_foreground - udf_background) * grid_size

        sdf_cost = np.exp(-(sdf - distance_threshold) / distance_scale_factor)

        dip_cost = np.clip(dip_data / max_dip, 0, 1)
        roughness_cost = np.clip(roughness_data / max_roughness, 0, 1)

        # set the cost to 1 where there is an overflow
        dip_cost[total_overflow] = 1
        roughness_cost[total_overflow] = 1

        dip_weight = 0.75
        roughness_weight = 0.25

        geometry_cost = dip_weight * dip_cost + roughness_weight * roughness_cost

        geometry_weight = 0.5
        sdf_weight = 0.5
        total_cost = geometry_weight * geometry_cost + sdf_weight * sdf_cost

        return total_cost, sdf_cost, geometry_cost

    def bake_lut(self, interp_method="bspline", inflation_radius: float = 1):
        self._inflation_radius = inflation_radius
        geotiff_folder = os.path.dirname(self._geotiff_reader.file_path)
        self.lut_file_path = os.path.join(
            geotiff_folder, f"cost_lut_for_xy_{interp_method}.casadi"
        )
        self._build_cost()
        if os.path.exists(self.lut_file_path):
            print(f"[bold yellow]Loading cost LUT from {self.lut_file_path}[/]")
            self._lut_for_xy = ca.Function.load(self.lut_file_path)
            return

        print(f"[bold yellow]Building cost LUT from {self.lut_file_path}[/]")
        self._bake_lut_for_row_col(interp_method)
        self._bake_lut_for_xy()

    def _build_cost(self):
        dip_data = self._geotiff_reader.read_dip_data()
        roughness_data = self._geotiff_reader.read_roughness_data()

        np.nan_to_num(dip_data, copy=False)
        np.nan_to_num(roughness_data, copy=False)

        inflation_pixels = int(self._inflation_radius / self._geotiff_reader.grid_size)

        total_cost, sdf_cost, geometry_cost = GeoTiffCostLut.cost_callable(
            dip_data, roughness_data, self._geotiff_reader.grid_size, inflation_pixels
        )
        self._cost = total_cost
        self._sdf_cost = sdf_cost
        self._geometry_cost = geometry_cost

    def _bake_lut_for_row_col(self, interp_method="bspline"):
        match interp_method:
            case "bspline":
                pass
            case "linear":
                pass
            case _:
                raise ValueError(f"Interpolation method {interp_method} not supported")

        row_grid, col_grid = (
            np.arange(self._geotiff_reader.height),
            np.arange(self._geotiff_reader.width),
        )
        self._lut_for_row_col = ca.interpolant(
            "lut_for_row_col",
            interp_method,
            [row_grid, col_grid],
            self._cost.ravel(order="F"),
        )

    def _bake_lut_for_xy(self):
        x, y = ca.MX.sym("x"), ca.MX.sym("y")
        row, col = self._geotiff_reader.xy_to_row_col(x, y)
        self._lut_for_xy = ca.Function(
            "lut_for_xy",
            [x, y],
            [self._lut_for_row_col(ca.vertcat(row, col))],
            ["x", "y"],
            ["cost"],
        )
        self._lut_for_xy.save(self.lut_file_path)

    @property
    def lut_for_xy(self) -> ca.Function:
        return self._lut_for_xy

    @lut_for_xy.setter
    def lut_for_xy(self, lut: ca.Function):
        self._lut_for_xy = lut

    @property
    def lut_for_row_col(self) -> ca.Function:
        return self._lut_for_row_col

    @property
    def cost(self):
        return self._cost

    @property
    def sdf_cost(self):
        return self._sdf_cost

    @property
    def geometry_cost(self):
        return self._geometry_cost

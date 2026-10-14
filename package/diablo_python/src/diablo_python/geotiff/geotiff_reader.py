import numpy as np
import rasterio
import rasterio.io
import rasterio.plot
from rasterio.transform import Affine


class GeoTiffReader:
    def __init__(self, file_path: str, dip_band: int, roughness_band: int):
        self.file_path = file_path
        self._dataset: rasterio.io.DatasetReader = rasterio.open(file_path)
        self._width: int = self._dataset.width
        self._height: int = self._dataset.height
        self._transform: Affine = self._dataset.transform
        self._grid_size: float = abs(self._transform.a)
        self.set_band(dip_band, roughness_band)

    def set_band(self, dip_band: int, roughness_band: int):
        self._dip_band = dip_band
        self._roughness_band = roughness_band

    def read_dip_data(self) -> np.ndarray:
        self._dip_data: np.ndarray = self._dataset.read(self._dip_band)
        return self._dip_data

    def read_roughness_data(self) -> np.ndarray:
        self._roughness_data: np.ndarray = self._dataset.read(self._roughness_band)
        return self._roughness_data

    def show_meta(self):
        print(self._dataset.meta)

    def show_stats(self):
        print(self._dataset.stats())

    def show_image(self):
        rasterio.plot.show(self._dataset)

    def read_data(self, band: int | None) -> np.ndarray:
        if band:
            return self._dataset.read(band)
        else:
            return self._dataset.read()

    def row_col_to_xy(self, row: int, col: int) -> tuple[float, float]:
        return self._dataset.xy(row, col)

    def xy_to_row_col(self, x: float, y: float) -> tuple[int, int]:
        tf_xy_inv = ~self._transform
        tf_xy_inv_corrected = Affine(
            tf_xy_inv.a,
            tf_xy_inv.b,
            tf_xy_inv.c - 0.5,
            tf_xy_inv.d,
            tf_xy_inv.e,
            tf_xy_inv.f - 0.5,
        )
        col, row = tf_xy_inv_corrected * (x, y)
        return row, col

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def grid_size(self):
        return self._grid_size

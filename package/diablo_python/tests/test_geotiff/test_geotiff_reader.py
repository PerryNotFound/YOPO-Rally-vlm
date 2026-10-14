import numpy as np
import pytest

from diablo_python.geotiff import GeoTiffReader


@pytest.mark.skip(reason="Show image will block the test")
def test_geotiff_show_image(geotiff_reader: GeoTiffReader):
    geotiff_reader.show_image()


def test_geotiff_reader_read_data(geotiff_reader: GeoTiffReader):
    geotiff_reader.show_meta()
    geotiff_reader.show_stats()
    geotiff_reader.set_band(2, 4)
    dip_data = geotiff_reader.read_dip_data()
    roughness_data = geotiff_reader.read_roughness_data()

    assert (
        dip_data.shape
        == roughness_data.shape
        == (geotiff_reader.height, geotiff_reader.width)
    )


@pytest.mark.parametrize(
    "row, col",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (300, 150),
        (150, 300),
        (300, 300),
    ],
)
def test_geotiff_reader_row_col_to_xy(row, col, geotiff_reader: GeoTiffReader):
    x, y = geotiff_reader.row_col_to_xy(row, col)
    row_, col_ = geotiff_reader.xy_to_row_col(x, y)
    assert np.isclose(row, row_), "row conversion failed"
    assert np.isclose(col, col_), "col conversion failed"


@pytest.mark.parametrize(
    "x, y",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (300, 150),
        (150, 300),
        (300, 300),
    ],
)
def test_geotiff_reader_xy_to_row_col(x, y, geotiff_reader: GeoTiffReader):
    row, col = geotiff_reader.xy_to_row_col(x, y)
    x_, y_ = geotiff_reader.row_col_to_xy(row, col)
    assert np.isclose(x, x_), "x conversion failed"
    assert np.isclose(y, y_), "y conversion failed"

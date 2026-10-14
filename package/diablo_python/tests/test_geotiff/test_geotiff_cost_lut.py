import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pytest

from diablo_python.geotiff import GeoTiffCostLut, GeoTiffReader


@pytest.fixture(scope="module")
def lut_for_row_col(lut_linear: GeoTiffCostLut):
    return lut_linear.lut_for_row_col


@pytest.fixture(scope="module")
def lut_for_xy(lut_linear: GeoTiffCostLut):
    return lut_linear.lut_for_xy


@pytest.mark.skip(reason="Show cost will block the test")
def test_geotiff_show_cost(lut_linear: GeoTiffCostLut):
    figure, ax = plt.subplots(2, 2, figsize=(10, 8))

    cax0 = ax[0, 0].imshow(lut_linear.cost)
    ax[0, 0].set_title("Cost")
    figure.colorbar(cax0, ax=ax[0, 0])

    cax1 = ax[0, 1].imshow(lut_linear.sdf_cost)
    ax[0, 1].set_title("SDF Cost")
    figure.colorbar(cax1, ax=ax[0, 1])

    cax2 = ax[1, 0].imshow(lut_linear.geometry_cost)
    ax[1, 0].set_title("Geometry Cost")
    figure.colorbar(cax2, ax=ax[1, 0])

    plt.tight_layout()
    plt.show()


@pytest.fixture(scope="module")
def get_interped_cost_xy(geotiff_reader: GeoTiffReader, lut_for_xy: ca.Function):
    interped_cost = np.zeros((geotiff_reader.height, geotiff_reader.width))
    for row in range(geotiff_reader.height):
        for col in range(geotiff_reader.width):
            x, y = geotiff_reader.row_col_to_xy(row, col)
            interped_cost[row, col] = lut_for_xy(x, y)
    return interped_cost


@pytest.fixture(scope="module")
def get_interped_cost_row_col(
    geotiff_reader: GeoTiffReader, lut_for_row_col: ca.Function
):
    interped_cost = np.zeros((geotiff_reader.height, geotiff_reader.width))
    for row in range(geotiff_reader.height):
        for col in range(geotiff_reader.width):
            interped_cost[row, col] = lut_for_row_col([row, col])
    return interped_cost


@pytest.mark.skip(reason="too slow")
def test_geotiff_cost_lut_for_row_col(
    get_interped_cost_row_col: np.ndarray, lut_linear: GeoTiffCostLut
):
    assert np.allclose(get_interped_cost_row_col, lut_linear.cost), "Cost not equal"


@pytest.mark.skip(reason="Too slow")
def test_geotiff_cost_lut_for_xy(
    get_interped_cost_xy: np.ndarray, lut_linear: GeoTiffCostLut
):
    assert np.allclose(get_interped_cost_xy, lut_linear.cost), "Cost not equal"


@pytest.mark.skip(reason="Show cost image will block the test")
def test_show_cost_xy(get_interped_cost_xy: np.ndarray, geotiff_reader: GeoTiffReader):
    bounds = geotiff_reader._dataset.bounds
    plt.imshow(
        get_interped_cost_xy,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    )
    plt.colorbar()
    plt.show()


@pytest.mark.skip(reason="Show cost image will block the test")
def test_show_cost_xy_3d(
    get_interped_cost_xy: np.ndarray, geotiff_reader: GeoTiffReader
):
    print(f"geotiff bounds: {geotiff_reader._dataset.bounds}")
    bounds = geotiff_reader._dataset.bounds
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = np.linspace(bounds.left, bounds.right, geotiff_reader.width)
    y = np.linspace(bounds.bottom, bounds.top, geotiff_reader.height)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, get_interped_cost_xy, cmap="viridis")
    plt.show()

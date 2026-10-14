import os
import time

from diablo_python.network.coord_transform import CoordTransform
import pytest
from torch.utils.data import DataLoader

from diablo_python.geotiff import GeoTiffCostLut, GeoTiffReader
from diablo_python.network import (
    YOPODataset,
    YOPOTrainTransform,
    StateTransform,
    OutputTransform,
)


@pytest.fixture(scope="session")
def test_data_folder_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    data_folder_path = os.path.join(dir_path, R"data")
    return data_folder_path


@pytest.fixture(scope="session")
def geotiff_reader(test_data_folder_path: str):
    dip_band = 2
    roughness_band = 4
    tiff_path = os.path.join(test_data_folder_path, R"testPCG.tif")
    return GeoTiffReader(tiff_path, dip_band, roughness_band)


@pytest.fixture(scope="session")
def lut_linear(geotiff_reader: GeoTiffReader):
    geotiff_cost_lut = GeoTiffCostLut(geotiff_reader)

    geotiff_cost_lut.bake_lut("linear")
    return geotiff_cost_lut


@pytest.fixture(scope="session")
def lut_bspline(geotiff_reader: GeoTiffReader):
    time_start = time.time()
    geotiff_cost_lut = GeoTiffCostLut(geotiff_reader)

    geotiff_cost_lut.bake_lut("bspline")
    time_end = time.time()
    print(f"Time to bake bspline: {time_end - time_start}")
    return geotiff_cost_lut


@pytest.fixture(scope="session")
def candidate_num():
    candidate_num = 5
    return candidate_num


@pytest.fixture(scope="session")
def train_data_folder_path(test_data_folder_path: str):
    train_data_folder_path = os.path.join(test_data_folder_path, R"test_data_process")
    # train_data_folder_path = R"D:\Unity Projects\Learn\PCG_Test\Assets\.MyData_Test"
    return train_data_folder_path


@pytest.fixture(scope="session")
def yopo_data_loader(train_data_folder_path: str, candidate_num: int):
    state_transform = StateTransform()
    output_transform = OutputTransform()
    transform = YOPOTrainTransform(
        state_transform=state_transform,
        output_transform=output_transform,
        coord_transform=CoordTransform(candidate_num=candidate_num),
    )
    yopo_dataset = YOPODataset(train_data_folder_path, transform, bake_cache=True)
    return DataLoader(yopo_dataset, batch_size=1, shuffle=False, pin_memory=True)

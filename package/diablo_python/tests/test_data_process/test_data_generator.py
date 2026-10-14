import pytest

from diablo_python.data_process import (
    CloudCompare,
    DataFileFinder,
    DataGenerator,
    TrajOpti,
)

tiff_file_prefix = "map"


@pytest.fixture
def data_file_finder(train_data_folder_path: str):
    data_file_finder = DataFileFinder(train_data_folder_path)
    return data_file_finder


@pytest.fixture
def data_generator(candidate_num: int, data_file_finder: DataFileFinder):
    cloud_compare = CloudCompare(tiff_file_prefix)
    traj_opt = TrajOpti(
        tiff_file_prefix,
        random_num_per_point=2,
        candidate_num=candidate_num,
        fixed_random_seed=0,
        multi_process=True,
    )
    data_generator = DataGenerator(data_file_finder)
    data_generator.setup_cloud_compare(cloud_compare)
    data_generator.setup_traj_opt(traj_opt)
    return data_generator


# @pytest.mark.skip(reason="This test is too slow to run on CI")
def test_data_generator(
    data_generator: DataGenerator, data_file_finder: DataFileFinder
):
    folders = data_file_finder.get_scene_folders()
    terrain_files, tree_files, toml_files, _ = data_file_finder.get_all_files()

    print("Folders:")
    for folder in folders:
        print(folder)

    print("\nTerrain Files:")
    for file in terrain_files:
        print(file)

    print("\nTree Files:")
    for file in tree_files:
        print(file)

    print("\nToml Files:")
    for file in toml_files:
        print(file)
    data_generator.process(True)


@pytest.mark.skip(reason="Visualize toml data will block the test")
def test_visualize_toml_data(
    data_generator: DataGenerator,
):
    data_generator.visualize_toml_opt_data(tiff_file_prefix)

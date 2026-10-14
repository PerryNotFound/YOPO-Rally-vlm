from diablo_python.data_process import (
    CloudCompare,
    DataFileFinder,
    DataGenerator,
    TrajOpti,
)

from diablo_project.configs import (
    test_data_folder_path,
    tiff_file_prefix,
    train_data_folder_path,
)


def generate_data_pipeline(
    tiff_file_prefix: str, train_data_folder_path: str, random_num_per_point=2
):
    data_file_finder = DataFileFinder(train_data_folder_path)
    cloud_compare = CloudCompare(tiff_file_prefix)
    traj_opt = TrajOpti(
        tiff_file_prefix,
        random_num_per_point=random_num_per_point,
        candidate_num=5,
        fixed_random_seed=0,
        multi_process=True,
    )
    data_generator = DataGenerator(data_file_finder)
    data_generator.setup_cloud_compare(cloud_compare)
    data_generator.setup_traj_opt(traj_opt)
    data_generator.process(True)


if __name__ == "__main__":
    generate_data_pipeline(tiff_file_prefix, train_data_folder_path, 100)
    generate_data_pipeline(tiff_file_prefix, test_data_folder_path, 20)

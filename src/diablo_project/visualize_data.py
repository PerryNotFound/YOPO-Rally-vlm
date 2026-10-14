from diablo_python.data_process import DataFileFinder, DataGenerator

from diablo_project.configs import tiff_file_prefix, train_data_folder_path

data_file_finder = DataFileFinder(train_data_folder_path)
data_generator = DataGenerator(data_file_finder)
data_generator.visualize_toml_opt_data(tiff_file_prefix)

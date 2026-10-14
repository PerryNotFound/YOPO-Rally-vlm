import os


YOPO_SIM_Path = "/home/admin0/YOPO-Sim"
load_model_path = "/home/admin0/YOPO-Rally/models/yopo_net.pth"
trained_model_folder = "models/temp"

train_data_folder_path = os.path.join(YOPO_SIM_Path, "TrainingData")
test_data_folder_path = os.path.join(YOPO_SIM_Path, "TestingData")

tiff_file_prefix = "map"

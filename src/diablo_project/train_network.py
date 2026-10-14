import os

import torch
from diablo_python.network import creata_yopo_trainer

from diablo_project.configs import train_data_folder_path, trained_model_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


yopo_trainer = creata_yopo_trainer(
    batch_size=64,
    data_folder_path=train_data_folder_path,
    device=device,
    epochs=20,
    model_folder=trained_model_folder,
    save_on_exit=True,
)


def train_yopo_model():
    yopo_trainer.train(
        checkpoint_path=os.path.join(trained_model_folder, "yopo_net_state.pth")
    )


if __name__ == "__main__":
    train_yopo_model()

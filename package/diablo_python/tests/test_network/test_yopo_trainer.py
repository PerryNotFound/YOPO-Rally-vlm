import os

import pytest
import torch

from diablo_python.network import creata_yopo_trainer


@pytest.mark.parametrize(
    "device",
    [torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.device("cpu")],
)
def test_yopo_trainer(
    test_data_folder_path: str, train_data_folder_path: str, device: torch.device
):
    log_dir = os.path.join(test_data_folder_path, "logs")
    model_folder = os.path.join(test_data_folder_path, "model")
    batch_size = 1

    yopo_trainer = creata_yopo_trainer(
        batch_size=batch_size,
        data_folder_path=train_data_folder_path,
        device=device,
        epochs=2,
        log_folder=log_dir,
        model_folder=model_folder,
        bake_cache=True,
    )
    yopo_trainer.train()
    yopo_trainer.save_model()

import torch
from diablo_python.network import (
    OutputTransform,
    StateTransform,
    YOPODataset,
    YOPOTrainTransform,
)
from diablo_python.network.coord_transform import CoordTransform
from diablo_python.network.yopo_trainer import YOPOTrainer
from torch.utils.data import DataLoader

from diablo_project.configs import load_model_path, test_data_folder_path


def test_yopo_model():
    test_dataset = YOPODataset(
        data_folder_path=test_data_folder_path,
        transform=YOPOTrainTransform(
            state_transform=StateTransform(),
            output_transform=OutputTransform(),
            coord_transform=CoordTransform(),
        ),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.jit.load(load_model_path).to(device).eval()
    net = torch.jit.optimize_for_inference(net)
    yopo_trainer = YOPOTrainer(
        test_loader, test_loader, net=net, device=device, save_on_exit=False
    )
    yopo_trainer.test_one_epoch(test_loader)


if __name__ == "__main__":
    test_yopo_model()

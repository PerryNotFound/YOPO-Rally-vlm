import atexit
import os
from datetime import datetime
from typing import Optional

import torch
from rich.progress import Progress
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from diablo_python.network.coord_transform import CoordTransform

from .yopo_dataset import YOPODataset
from .yopo_net import YOPONet
from .yopo_transforms import OutputTransform, StateTransform, YOPOTrainTransform


class YOPOTrainer:
    def __init__(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        net: YOPONet | torch.jit.ScriptModule,
        device: torch.device,
        epochs: int = 20,
        log_folder: str = "logs",
        model_folder: str = "models",
        amp_enabled: bool = True,
        save_on_exit: bool = True,
    ):
        self._training_loader = training_loader
        self._validation_loader = validation_loader
        self._net = net
        self._device = device
        self._epochs = epochs
        self._amp_enabled = amp_enabled

        self._loss_fn = nn.MSELoss()

        timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        LOG_DIR = os.path.join(log_folder, f"YOPO_{timestamp}")
        self._writer = SummaryWriter(log_dir=LOG_DIR)

        self._progress = Progress()
        self._model_folder = model_folder
        if save_on_exit:
            self._exit_func = atexit.register(self.save_model)

    def train(self, checkpoint_path: str = ""):
        self._opt = torch.optim.AdamW(self._net.parameters(), lr=1e-3, fused=True)
        self._scaler = GradScaler(enabled=self._amp_enabled)
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self._net.load_state_dict(state_dict)
            self._progress.console.log("Checkpoint loaded successfully")
        except FileNotFoundError:
            self._progress.console.log("No checkpoint found, training from scratch")
        with self._progress:
            total_progress = self._progress.add_task("Training", total=self._epochs)
            for epoch in range(self._epochs):
                self._net.train()
                avg_loss = self._train_one_epoch(epoch, total_progress)
                self._eval_one_epoch(epoch, avg_loss)

    def save_model(self, file_name="yopo_net.pth"):
        self._progress.console.log("Saving model...")
        file_path = os.path.join(self._model_folder, file_name)

        if not os.path.exists(self._model_folder):
            os.makedirs(self._model_folder)

        net = torch.jit.script(self._net)
        torch.jit.save(net, file_path)
        torch.save(
            self._net.state_dict(),
            os.path.join(self._model_folder, "yopo_net_state.pth"),
        )
        atexit.unregister(self._exit_func)

    def _train_one_epoch(self, epoch: int, total_progress: int) -> Tensor:
        one_epoch_progress = self._progress.add_task(
            f"Epoch: {epoch}", total=len(self._training_loader)
        )

        running_loss, last_loss = 0, 0
        for batch_idx, (image, state, output) in enumerate(self._training_loader):
            state: Tensor = state.to(self._device)
            image: Tensor = image.to(self._device)
            output: Tensor = output.to(self._device)
            with autocast(device_type=self._device.type, enabled=self._amp_enabled):
                pred = self._net(image, state)
                loss = self._loss_fn(pred, output)
            self._scaler.scale(loss).backward()
            self._scaler.step(self._opt)
            self._scaler.update()

            self._progress.update(one_epoch_progress, advance=1)
            self._progress.update(
                total_progress, advance=1 / len(self._training_loader)
            )

            running_loss += loss
            inspect_interval = max(1, len(self._training_loader) // 16)
            if batch_idx % inspect_interval == inspect_interval - 1:
                last_loss = running_loss / inspect_interval
                self._progress.console.log(
                    f"Epoch: {epoch}, Loss: {last_loss.item():.4g}"
                )
                self._writer.add_scalar(
                    "Loss/Train",
                    last_loss.item(),
                    epoch * len(self._training_loader) + batch_idx,
                )
                self._writer.flush()
                running_loss = 0
        self._progress.remove_task(one_epoch_progress)
        return last_loss

    @torch.inference_mode()
    def _eval_one_epoch(self, epoch: int, avg_loss: Tensor):
        running_v_total_loss = 0
        running_v_cost_loss = 0
        running_v_velocity_loss = 0
        running_v_position_norm_loss = 0
        running_v_position_theta_loss = 0
        for _batch_idx, (vimage, vstate, voutput) in enumerate(self._validation_loader):
            vstate: Tensor = vstate.to(self._device)
            vimage: Tensor = vimage.to(self._device)
            voutput: Tensor = voutput.to(self._device)
            with autocast(device_type=self._device.type, enabled=self._amp_enabled):
                pred = self._net(vimage, vstate)

                running_v_total_loss += self._loss_fn(pred, voutput)
                running_v_cost_loss += self._loss_fn(
                    pred[:, -1, ...], voutput[:, -1, ...]
                )
                running_v_velocity_loss += self._loss_fn(
                    pred[:, 2:4, ...], voutput[:, 2:4, ...]
                )
                running_v_position_norm_loss += self._loss_fn(
                    pred[:, 0, ...], voutput[:, 0, ...]
                )
                running_v_position_theta_loss += self._loss_fn(
                    pred[:, 1, ...], voutput[:, 1, ...]
                )

        avg_v_total_loss = running_v_total_loss / len(self._validation_loader)
        avg_v_cost_loss = running_v_cost_loss / len(self._validation_loader)
        avg_v_velocity_loss = running_v_velocity_loss / len(self._validation_loader)
        avg_v_position_norm_loss = running_v_position_norm_loss / len(
            self._validation_loader
        )
        avg_v_position_theta_loss = running_v_position_theta_loss / len(
            self._validation_loader
        )

        self._progress.console.log(f"Validation Total Loss: {avg_v_total_loss:.4g}")

        self._writer.add_scalars(
            "Loss", {"Validation": avg_v_total_loss, "Training": avg_loss}, epoch
        )
        self._writer.add_scalars(
            "Loss/Validation/Detail",
            {
                "Cost": avg_v_cost_loss,
                "Velocity": avg_v_velocity_loss,
                "Position norm": avg_v_position_norm_loss,
                "Position theta": avg_v_position_theta_loss,
            },
            epoch,
        )
        self._writer.flush()

    @torch.inference_mode()
    def test_one_epoch(
        self,
        test_loader: DataLoader,
    ):
        running_t_total_loss = 0
        running_t_cost_loss = 0
        running_t_velocity_loss = 0
        running_t_position_norm_loss = 0
        running_t_position_theta_loss = 0
        for _batch_idx, (vimage, vstate, voutput) in enumerate(test_loader):
            vstate: Tensor = vstate.to(self._device)
            vimage: Tensor = vimage.to(self._device)
            voutput: Tensor = voutput.to(self._device)
            pred = self._net(vimage, vstate)

            running_t_total_loss += self._loss_fn(pred, voutput)
            running_t_cost_loss += self._loss_fn(pred[:, -1, ...], voutput[:, -1, ...])
            running_t_velocity_loss += self._loss_fn(
                pred[:, 2:4, ...], voutput[:, 2:4, ...]
            )
            running_t_position_norm_loss += self._loss_fn(
                pred[:, 0, ...], voutput[:, 0, ...]
            )
            running_t_position_theta_loss += self._loss_fn(
                pred[:, 1, ...], voutput[:, 1, ...]
            )

        avg_v_total_loss = running_t_total_loss / len(self._validation_loader)
        avg_v_cost_loss = running_t_cost_loss / len(self._validation_loader)
        avg_v_velocity_loss = running_t_velocity_loss / len(self._validation_loader)
        avg_v_position_norm_loss = running_t_position_norm_loss / len(
            self._validation_loader
        )
        avg_v_position_theta_loss = running_t_position_theta_loss / len(
            self._validation_loader
        )

        self._progress.console.log(f"Test Total Loss: {avg_v_total_loss:.4g}")
        self._progress.console.log(f"Test Cost Loss: {avg_v_cost_loss:.4g}")
        self._progress.console.log(f"Test Velocity Loss: {avg_v_velocity_loss:.4g}")
        self._progress.console.log(
            f"Test Position Norm Loss: {avg_v_position_norm_loss:.4g}"
        )
        self._progress.console.log(
            f"Test Position Theta Loss: {avg_v_position_theta_loss:.4g}"
        )


def creata_yopo_trainer(
    batch_size: int,
    data_folder_path: str,
    device: torch.device,
    epochs: int,
    random_seed: Optional[int] = 0,
    amp_enabled=True,
    save_on_exit=True,
    bake_cache=True,
    log_folder="logs",
    model_folder="models",
):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    transform = YOPOTrainTransform(
        state_transform=StateTransform(),
        output_transform=OutputTransform(),
        coord_transform=CoordTransform(),
    )
    dataset = YOPODataset(
        data_folder_path=data_folder_path,
        transform=transform,
        image_size=(160, 32),
        bake_cache=bake_cache,
    )

    training_set, validation_set = random_split(dataset, [0.9, 0.1])
    training_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    net = YOPONet().to(device)
    yopo_trainer = YOPOTrainer(
        training_loader=training_loader,
        validation_loader=validation_loader,
        net=net,
        device=device,
        epochs=epochs,
        amp_enabled=amp_enabled,
        save_on_exit=save_on_exit,
        log_folder=log_folder,
        model_folder=model_folder,
    )

    return yopo_trainer

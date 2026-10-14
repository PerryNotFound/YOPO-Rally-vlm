import pytest
import torch
from torch import Tensor

from diablo_python.network import YOPONet


@pytest.mark.parametrize(
    "device",
    [torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.device("cpu")],
)
@pytest.mark.parametrize("batch_num", [32, 64])
@torch.inference_mode()
def test_yopo_net_batch(batch_num: int, device: torch.device):
    yopo_net = YOPONet().to(device)
    yopo_net = torch.jit.script(yopo_net)
    yopo_net = torch.jit.optimize_for_inference(yopo_net)
    test_image = torch.rand((batch_num, 1, 160, 32), device=device)
    test_state = torch.rand((batch_num, 4, 5, 1), device=device)

    output: Tensor = yopo_net(test_image, test_state)
    p_n_hat, p_theta_hat, v_x_hat, v_y_hat, cost = output.unbind(dim=1)

    assert torch.all((p_n_hat >= 0) & (p_n_hat <= 1)), "p_n_hat is not in [0, 1]"
    assert torch.all(
        (p_theta_hat >= -1) & (p_theta_hat <= 1)
    ), "p_theta_hat is not in [-1, 1]"
    assert torch.all((v_x_hat >= -1) & (v_x_hat <= 1)), "v_x_hat is not in [-1, 1]"
    assert torch.all((v_y_hat >= -1) & (v_y_hat <= 1)), "v_y_hat is not in [-1, 1]"
    assert torch.all((cost >= 0) & (cost <= 1)), "cost is not in [0, 1]"

    assert output.shape[1:] == (5, 5, 1), "Output shape is not correct"

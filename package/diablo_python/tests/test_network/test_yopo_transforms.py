import torch
from torch.utils.data import DataLoader

from diablo_python.network import CoordTransform, OutputTransform


@torch.inference_mode()
def test_transform_data(
    yopo_data_loader: DataLoader,
):
    for _index, (_image, state_hat, output_hat) in enumerate(yopo_data_loader):
        vx_hat, vy_hat, gx_hat, gy_hat = state_hat.unbind(dim=1)
        pn_hat, ptheta_hat, vx_hat, vy_hat, cost_hat = output_hat.unbind(dim=1)

        assert torch.all((vx_hat >= -1) & (vx_hat <= 1)), "vx_hat is not in [-1, 1]"
        assert torch.all((vy_hat >= -1) & (vy_hat <= 1)), "vy_hat is not in [-1, 1]"
        assert torch.all((gx_hat >= -1) & (gx_hat <= 1)), "gx_hat is not in [-1, 1]"
        assert torch.all((gy_hat >= -1) & (gy_hat <= 1)), "gy_hat is not in [-1, 1]"

        assert torch.all((pn_hat >= 0) & (pn_hat <= 1)), "px_hat is not in [0, 1]"
        assert torch.all(
            (ptheta_hat >= -1) & (ptheta_hat <= 1)
        ), "ptheta_hat is not in [-1, 1]"


def test_output_transform():
    candidate_num = 6
    output_transform = OutputTransform()
    coord_transform = CoordTransform(candidate_num=candidate_num)
    pos_end_w = torch.rand(2, candidate_num)
    vel_end_w = torch.rand(2, candidate_num)
    cost = torch.rand(1, candidate_num)
    yaw_w = torch.rand(1)
    pos_w = torch.rand(2)
    coord_transform.construct_primitive_frame(yaw_w, pos_w)

    T_wp = coord_transform.T_wp
    T_pw = coord_transform.T_pw
    output: torch.Tensor = output_transform(T_wp, pos_end_w, vel_end_w, cost)

    pos_end_w_test, vel_end_w_test, cost_test = output_transform.inverse_transform(
        output, T_pw
    )

    assert torch.allclose(
        pos_end_w, pos_end_w_test, atol=1e-6
    ), "pos_end_w is not equal to pos_end_w_test"
    assert torch.allclose(
        vel_end_w, vel_end_w_test, atol=1e-6
    ), "vel_end_w is not equal to vel_end_w_test"
    assert torch.allclose(cost, cost_test, atol=1e-6), "cost is not equal to cost_test"

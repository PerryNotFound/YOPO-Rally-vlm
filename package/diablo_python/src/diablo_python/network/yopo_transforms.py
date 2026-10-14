import numpy as np
import torch
from torch import Tensor, nn

from .coord_transform import CoordTransform


class YOPOTrainTransform(nn.Module):
    """Transform `yaw_w`, `pos_start_w`, `vel_start_w`, `goal_w`, `pos_end_w`, `vel_end_w`, `cost` to
    `vx_start_p`, `vy_start_p`, `gx_p`, `gy_p` and
    `pn_end_p`, `pt_end_p`, `vx_end_p`, `vy_end_p`, `cost` from world frame to primitive frame.

    Normalize `vx_start_p`, `vy_start_p`, `gx_p`, `gy_p` to [-1, 1] range, then concatenate them in channel dimension.
    Normalize the `pn_end_p`, `pt_end_p`, `vx_end_p`, `vy_end_p`, `cost` to [-1, 1] range, then concatenate them in channel dimension.

    `yaw_w` shape = (1,)
    `pos_start_w`, `vel_start_w`, `goal_w` shape: (2,)
    `pos_end_w`, `vel_end_w` shape: (2, W)

    `vx_start_p`, `vy_start_p`, `gx_p`, `gy_p` shape: (1, H, W)
    `state` shape: (4, H, W)

    `pn_end_p`, `pt_end_p`, `vx_end_p`, `vy_end_p`, cost shape: (1, H, W)
    `output` shape: (5, H, W)
    """

    def __init__(
        self,
        state_transform: "StateTransform",
        output_transform: "OutputTransform",
        coord_transform: "CoordTransform",
    ):
        super().__init__()
        self._state_transform = state_transform
        self._output_transform = output_transform
        self._coord_transform = coord_transform
        self._candidate_num = self._coord_transform.candidate_num

    def forward(
        self,
        yaw_w: Tensor,
        pos_start_w: Tensor,
        vel_start_w: Tensor,
        goal_w: Tensor,
        pos_end_w: Tensor,
        vel_end_w: Tensor,
        cost: Tensor,
    ):
        assert yaw_w.shape == (1,)
        assert pos_start_w.shape == vel_start_w.shape == goal_w.shape == (2,)
        assert pos_end_w.shape == vel_end_w.shape == (2, self._candidate_num)
        assert cost.shape == (1, self._candidate_num)

        self._coord_transform.construct_primitive_frame(yaw_w, pos_start_w)

        state: Tensor = self._state_transform(
            self._coord_transform.T_wp, vel_start_w, goal_w
        )
        output: Tensor = self._output_transform(
            self._coord_transform.T_wp, pos_end_w, vel_end_w, cost
        )
        return state, output


class YOPOEvalTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        yaw_w: Tensor,
        pos_start_w: Tensor,
        vel_start_w: Tensor,
        goal_w: Tensor,
        pos_end_w: Tensor,
        vel_end_w: Tensor,
        cost: Tensor,
    ):
        return yaw_w, pos_start_w, vel_start_w, goal_w, pos_end_w, vel_end_w, cost


class StateTransform(nn.Module):
    def __init__(self, vel_max=3.0, goal_max=15.0):
        super().__init__()
        self._vel_max = vel_max
        self._goal_max = goal_max

    def forward(
        self,
        T_wp: Tensor,
        vel_start_w: Tensor,
        goal_w: Tensor,
    ):
        vel_start_w_candidate = vel_start_w.unsqueeze(0)
        goal_w_candidate = goal_w.unsqueeze(0)
        vel_start_p = CoordTransform.transform_to_frame(
            T_wp, vel_start_w_candidate, False
        )
        goal_p = CoordTransform.transform_to_frame(T_wp, goal_w_candidate)
        state = self._make_state(vel_start_p, goal_p)
        return state

    def _make_state(self, vel_start_p: Tensor, goal_p: Tensor):
        vel_start_p_hat = vel_start_p / self._vel_max
        goal_p_hat = goal_p / self._goal_max
        state = torch.cat([vel_start_p_hat, goal_p_hat], dim=-1).unsqueeze(0)

        # (4, H, W)
        state = state.permute(2, 0, 1)
        return state


class OutputTransform(nn.Module):
    def __init__(
        self, vel_max=3.0, pn_max=15.0, pt_max=np.pi / 3, cost_max=50.0
    ) -> None:
        super().__init__()
        self._vel_max = vel_max
        self._pn_max = pn_max
        self._pt_max = pt_max
        self._cost_max = cost_max

    def forward(
        self,
        T_wp: Tensor,
        pos_end_w: Tensor,
        vel_end_w: Tensor,
        cost: Tensor,
    ):
        pos_end_w_candidate = pos_end_w.transpose(0, 1)
        vel_end_w_candidate = vel_end_w.transpose(0, 1)
        pos_end_p = CoordTransform.transform_to_frame(T_wp, pos_end_w_candidate)
        vel_end_p = CoordTransform.transform_to_frame(T_wp, vel_end_w_candidate, False)
        output = self._make_output(pos_end_p, vel_end_p, cost)
        return output

    def _make_output(self, pos_end_p: Tensor, vel_end_p: Tensor, cost: Tensor):
        pos_end_p_norm = torch.norm(pos_end_p, dim=-1, keepdim=True)
        pos_end_p_theta = torch.atan2(pos_end_p[..., 1], pos_end_p[..., 0]).unsqueeze(
            -1
        )
        pos_end_pn_hat = pos_end_p_norm / self._pn_max
        pos_end_pt_hat = pos_end_p_theta / self._pt_max
        vel_end_p_hat = vel_end_p / self._vel_max
        cost_hat = cost / self._cost_max
        cost_hat = cost_hat.T

        output = torch.cat(
            [pos_end_pn_hat, pos_end_pt_hat, vel_end_p_hat, cost_hat], dim=-1
        ).unsqueeze(0)
        # (5, H, W)
        output = output.permute(2, 0, 1)
        return output

    @torch.inference_mode()
    def inverse_transform(self, output: Tensor, T_pw: Tensor):
        pos_end_pn_hat = output[0, :, :]
        pos_end_pt_hat = output[1, :, :]
        vel_end_p_hat = output[2:4, :, :]
        cost_hat = output[4, :, :]

        pos_end_pn = pos_end_pn_hat * self._pn_max
        pos_end_pt = pos_end_pt_hat * self._pt_max
        vel_end_p = vel_end_p_hat * self._vel_max
        cost = cost_hat * self._cost_max

        pos_end_p_norm = pos_end_pn
        pos_end_p_theta = pos_end_pt
        pos_end_p = torch.stack(
            [
                pos_end_p_norm * torch.cos(pos_end_p_theta),
                pos_end_p_norm * torch.sin(pos_end_p_theta),
            ],
            dim=-1,
        )
        pos_end_p = pos_end_p.squeeze()
        vel_end_p = vel_end_p.squeeze().transpose(0, 1)

        pos_end_w_candidate = CoordTransform.transform_to_frame(T_pw, pos_end_p)
        pos_end_w = pos_end_w_candidate.transpose(0, 1)
        vel_end_candidate = CoordTransform.transform_to_frame(T_pw, vel_end_p, False)
        vel_end_w = vel_end_candidate.transpose(0, 1)
        return pos_end_w, vel_end_w, cost

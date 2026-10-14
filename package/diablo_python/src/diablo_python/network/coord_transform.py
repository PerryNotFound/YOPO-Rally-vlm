import numpy as np
import torch
from torch import Tensor


class CoordTransform:
    def __init__(self, candidate_num=5, camera_h_fov_rad=np.pi / 2) -> None:
        self._camera_h_fov_rad = camera_h_fov_rad
        self.candidate_num = candidate_num
        pass

    def construct_primitive_frame(self, yaw_w: Tensor, pos_start_w: Tensor):
        pass
        # The camera image is from left to right, so the theta should be clockwise from the center
        yaw_w_candidate = torch.linspace(
            self._camera_h_fov_rad / 2,
            -self._camera_h_fov_rad / 2,
            self.candidate_num + 1,
        ).unsqueeze(1)
        yaw_w_candidate = (yaw_w_candidate[:-1] + yaw_w_candidate[1:]) / 2 + yaw_w
        pos_w_candidate = pos_start_w.unsqueeze(0)
        self._T_pw = CoordTransform.construct_frame(pos_w_candidate, yaw_w_candidate)

        self._T_wp = CoordTransform.transform_fast_inverse(self._T_pw)

    @property
    def T_wp(self):
        return self._T_wp

    @property
    def T_pw(self):
        return self._T_pw

    @staticmethod
    def transform_fast_inverse(T: Tensor):
        assert T.shape[-2:] == (3, 3)
        R = T[..., :2, :2]
        t = T[..., :2, 2:]

        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat(
            [
                torch.cat([R_inv, t_inv], dim=-1),
                T[:, 2:3, :],
            ],
            dim=-2,
        )
        return T_inv

    @staticmethod
    def construct_frame(pos_w: Tensor, yaw_w: Tensor):
        candidate_num = yaw_w.shape[0]
        assert pos_w.shape[1:] == (2,)
        assert yaw_w.shape[1:] == (1,)

        # T_bw.shape = (candidate_num, 3, 3)
        T_bw = torch.eye(3).repeat(candidate_num, 1, 1)
        cos_yaw = torch.cos(yaw_w).squeeze(-1)
        sin_yaw = torch.sin(yaw_w).squeeze(-1)
        T_bw[:, 0, 0] = cos_yaw
        T_bw[:, 0, 1] = -sin_yaw
        T_bw[:, 1, 0] = sin_yaw
        T_bw[:, 1, 1] = cos_yaw

        T_bw[:, 0, 2] = pos_w[:, 0]
        T_bw[:, 1, 2] = pos_w[:, 1]

        return T_bw

    @staticmethod
    def transform_to_frame(
        T_wb: Tensor, coord_w: Tensor, apply_translation: bool = True
    ):
        # Return coord_b.shape = (-1, 2)
        assert T_wb.shape[1:] == (3, 3)
        assert coord_w.shape[1:] == (2,)

        # (candidate_num, 2, 1)
        coord_w = coord_w.unsqueeze(-1)
        if apply_translation:
            coord_w_homogeneous = torch.cat(
                [coord_w, torch.ones_like(coord_w[:, :1, :])], dim=1
            )
            # (candidate_num, 2)
            coord_b = torch.matmul(T_wb, coord_w_homogeneous).squeeze(-1)[:, :2]
            return coord_b
        else:
            coord_b = torch.matmul(T_wb[:, :2, :2], coord_w).squeeze(-1)
            # (candidate_num, 2)
            return coord_b

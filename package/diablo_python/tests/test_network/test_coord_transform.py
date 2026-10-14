import numpy as np
import torch
from torch import Tensor

from diablo_python.network.coord_transform import CoordTransform


@torch.inference_mode()
def test_coord_transform():
    pos_w = Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    yaw_w = Tensor([[np.pi / 2], [np.pi], [np.pi / 2], [0], [np.pi / 4]])
    T_wb = CoordTransform.construct_frame(pos_w, yaw_w)
    T_bw = CoordTransform.transform_fast_inverse(T_wb)

    test_w = Tensor([[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

    test_b = CoordTransform.transform_to_frame(T_bw, test_w)

    assert torch.allclose(
        test_w, CoordTransform.transform_to_frame(T_wb, test_b), atol=1e-6
    ), "Transformed back and forth should be the same"


@torch.inference_mode()
def test_coord_transform_random(
    candidate_num: int,
):
    pos_w = torch.rand(candidate_num, 2)
    yaw_w = torch.rand(candidate_num, 1)
    T_wb = CoordTransform.construct_frame(pos_w, yaw_w)
    T_bw = CoordTransform.transform_fast_inverse(T_wb)

    test_w = torch.rand(candidate_num, 2)

    test_b = CoordTransform.transform_to_frame(T_bw, test_w)

    assert torch.allclose(
        test_w, CoordTransform.transform_to_frame(T_wb, test_b), atol=1e-6
    ), "Transformed back and forth should be the same"

import torch
from torch import Tensor, nn

from .universal_blocks import CBL, CBCustomActivation, CustomResNet


class YOPOImageBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.custom_resnet = CustomResNet(image_channels=1, resnet_type="resnet18")

        self.cbl1 = CBL(in_channels=512, out_channels=128, stride=1)

    def forward(self, x: Tensor):
        x = self.custom_resnet(x)
        x = self.cbl1(x)
        return x


class YOPOStateBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cbl = CBL(in_channels=4, out_channels=32, kernel_size=1)

    def forward(self, x: Tensor):
        return self.cbl(x)


class YOPOBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = YOPOImageBranch()
        self.state_branch = YOPOStateBranch()

    def forward(self, image: Tensor, state: Tensor):
        image_features = self.image_branch(image)
        state_features = self.state_branch(state)
        x = torch.cat([image_features, state_features], dim=1)
        return x


class YOPOHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbl1 = CBL(in_channels=160, out_channels=512)
        self.cbl2 = CBL(in_channels=512, out_channels=256)
        # p_n^hat   p_theta^hat   v_x^hat   v_y^hat    cost
        # [0, 1]    [-1, 1]       [-1, 1]   [-1, 1]    [0, 1]
        self.cbl3 = CBCustomActivation(
            in_channels=256,
            out_channels=5,
            act=nn.ModuleList([nn.Sigmoid()] + [nn.Tanh()] * 3 + [nn.Sigmoid()]),
        )

    def forward(self, x: Tensor):
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cbl3(x)
        return x

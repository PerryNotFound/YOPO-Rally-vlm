from typing_extensions import override

import torch
from torch import Tensor, nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)


class CBL(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        act=nn.ReLU(),
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CBCustomActivation(CBL):
    def __init__(
        self,
        in_channels,
        out_channels,
        act: nn.ModuleList,
        kernel_size=3,
        stride=1,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        assert (
            len(act) == out_channels
        ), "Number of activation functions must match number of output channels"
        self.act = act

    @override
    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.stack([func(x[:, i, :, :]) for i, func in enumerate(self.act)], dim=1)
        return x


class CustomResNet(nn.Module):
    def __init__(self, image_channels=1, resnet_type="resnet18", pre_trained=True):
        super().__init__()
        match resnet_type:
            case "resnet18":
                original_resnet = resnet18(
                    weights=ResNet18_Weights.DEFAULT if pre_trained else None
                )
            case "resnet34":
                original_resnet = resnet34(
                    weights=ResNet34_Weights.DEFAULT if pre_trained else None
                )
            case _:
                raise ValueError("Unsupported resnet type")

        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

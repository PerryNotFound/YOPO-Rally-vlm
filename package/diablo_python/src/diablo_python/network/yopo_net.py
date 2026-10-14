from torch import Tensor, nn

from .blocks.yopo_blocks import YOPOBackBone, YOPOHead


class YOPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = YOPOBackBone()
        self.head = YOPOHead()

    def forward(self, image: Tensor, state: Tensor):
        x = self.backbone(image, state)
        x = self.head(x)
        return x

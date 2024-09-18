from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

# modified ResNet code by Chanho
# original code from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_groups: int = 2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes) if norm_layer is nn.BatchNorm2d else norm_layer(norm_groups, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes) if norm_layer is nn.BatchNorm2d else norm_layer(norm_groups, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        # Chanho: we don't need this for now
        raise NotImplementedError


class CustomResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        norm_groups: int = 2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        assert norm_layer is nn.BatchNorm2d or norm_layer is nn.GroupNorm, (
            "Chanho: only batchnorm and groupnorm are supported for now")
        self._norm_layer = norm_layer
        self._norm_groups = norm_groups

        self.inplanes = 8
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes) if norm_layer is nn.BatchNorm2d else norm_layer(norm_groups, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_layers = nn.ModuleList()
        for i in range(len(layers)):
            stride = 1 if i == 0 else 2
            block_num, channel_dim = layers[i]
            all_convs = self._make_layer(block, channel_dim, block_num, stride=stride)
            for conv in all_convs:
                self.conv_layers.append(conv)

        last_channel_dim = layers[-1][1]	
        self.reduce_resolution = self._reduce_spatial_resolution_with_3x3conv(last_channel_dim)
        self.fc1 = nn.Linear(1024, 256)
        # self.fc2 = nn.Linear(256, 32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm_layer is nn.BatchNorm2d:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            elif norm_layer is nn.GroupNorm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(self._norm_groups, planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, norm_layer=norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, norm_layer=norm_layer,
                )
            )

        return layers

    def _reduce_spatial_resolution_with_3x3conv(self, channel_dim):	
        if self._norm_layer is nn.BatchNorm2d:	
            bn = self._norm_layer(channel_dim) 	
        else:	
            bn = self._norm_layer(self._norm_groups, channel_dim)	
        net = nn.Sequential(	
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=2, padding=1, bias=False),	
            bn,	
            nn.ReLU(inplace=True)	
        )	
        return net

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        # print(f"conv1 shape: {x.shape}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f"maxpool shape: {x.shape}")
        for layer in self.conv_layers:
            x = layer(x)
            # print(f"conv shape: {x.shape}")
        x = self.reduce_resolution(x)
        # print(f"reduced shape: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"flattened shape: {x.shape}")
        x = self.fc1(x)
        # x = self.fc2(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _base_forward(self, x):
        """Handle batch of trajectories or model forward.
        [num steps per episode, num episode, num state per step]
        """
        size = x.size()
        dim = len(size)
        x = x.reshape(-1, 1, 128, 128)
        x = self.forward(x)
        if dim == 3:
            x = x.reshape(size[0], size[1], -1)
        elif dim == 1:
            x = x.squeeze()
        else:
            raise RuntimeError(f"Invalid input shape: {size}")
        return x

def count_parameters(model):
    num = 0
    for p in model.parameters():
        if p.requires_grad:
            num += p.numel()
    return num

if __name__ == "__main__":
    # Each tuple in the list below represents a setting for resnet layers (i.e., conv2_x. conv3_x, ... in Table 1 of the ResNet paper).
    # The first element in the tuple represents the number of repetition of the block in the correspending layer.
    # The second element in the tuple represents the number of channels used in the block.
    # If you compare the list input below with ResNet18 in Table 1 of the ResNet paper, you will be able to understand it more easily.

    # example 1: ResNet18 in Table 1 of the ResNet paper
    # (Technically it's not exactly the same as I removed the last 1000d fc layer and avg pooling)
    # model = CustomResNet(BasicBlock, [(2, 64), (2, 128), (2, 256), (2, 512)])

    # example 2: You can use it with group norm instead of batch norm, too
    # model = CustomResNet(BasicBlock, [(2, 64), (2, 128), (2, 256), (2, 512)], norm_layer=torch.nn.GroupNorm)

    # example 3: shallower ResNet that may be good enough for our case
    # model = CustomResNet(BasicBlock, [(2, 64), (2, 128)])

    # example 4: shallower ResNet with group norm that may be good enough for our case
    model = CustomResNet(BasicBlock, [(2, 8), (2, 16)], norm_layer=torch.nn.GroupNorm)

    print(count_parameters(model))
    img = torch.zeros(128*128)
    print(f"input shape: {img.shape}")
    out = model._base_forward(img)
    print(out.shape)
    img = torch.zeros(200, 10, 128*128)
    print(f"input shape: {img.shape}")
    out = model._base_forward(img)
    print(out.shape)
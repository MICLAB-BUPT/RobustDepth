import torch
from models.nonLocal import NONLocalBlock2D
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ShortcutProjection(nn.Module):
    """
    ## Linear projections for shortcut connection
    This does the $W_s x$ projection described above.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$
        * `out_channels` is the number of channels in $\mathcal{F}(x, \{W_i\})$
        * `stride` is the stride length in the convolution operation for $F$.
        We do the same stride on the shortcut connection, to match the feature-map size.
        """
        super().__init__()

        # Convolution layer for linear projection $W_s x$
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        # Paper suggests adding batch normalization after each convolution operation
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        # Convolution and batch normalization
        return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    """
    <a id="residual_block"></a>
    ## Residual Block
    This implements the residual block described in the paper.
    It has two $3 \times 3$ convolution layers.
    ![Residual Block](residual_block.svg)
    The first convolution layer maps from `in_channels` to `out_channels`,
    where the `out_channels` is higher than `in_channels` when we reduce the
    feature map size with a stride length greater than $1$.
    The second convolution layer maps from `out_channels` to `out_channels` and
    always has a stride length of 1.
    Both convolution layers are followed by batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$
        * `out_channels` is the number of output channels
        * `stride` is the stride length in the convolution operation.
        """
        super().__init__()

        # First $3 \times 3$ convolution layer, this maps to `out_channels`
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        # First activation function (ReLU)
        self.act1 = nn.ReLU()

        # Second $3 \times 3$ convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Batch normalization after the second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection should be a projection if the stride length is not $1$
        # of if the number of channels change
        if stride != 1 or in_channels != out_channels:
            # Projection $W_s x$
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            # Identity $x$
            self.shortcut = nn.Identity()

        # Second activation function (ReLU) (after adding the shortcut)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input of shape `[batch_size, in_channels, height, width]`
        """
        # Get the shortcut connection
        shortcut = self.shortcut(x)
        # First convolution and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Second convolution
        x = self.bn2(self.conv2(x))
        # Activation function after adding the shortcut
        return self.act2(x + shortcut)

class UncertaintyBlock(nn.Module):
    def __init__(self, img_input, p_input, n):
        super(UncertaintyBlock, self).__init__()
        self.rs1 = ResidualBlock(img_input, p_input, 1)
        list_block = []
        for i in range(n):
            list_block.append(ResidualBlock(p_input, p_input, 1))
        self.rs2 = nn.ModuleList(list_block)
        # 内容应该不同
        # self.rs3 = ResidualBlock(p_input * (n + 1), p_input, 1)
        self.nlb = NONLocalBlock2D(p_input * (n + 1))
        # self.seb = BasicBlock(p_input * (n + 1), p_input)
        list_block = []
        for i in range(n):
            list_block.append(ResidualBlock(p_input * (n + 2), p_input, 1))
        self.rs4 = nn.ModuleList(list_block)

    def forward(self, x, p):
        x = self.rs1(x)
        for index, m in enumerate(self.rs2):
            p[index] = m(p[index])
        c = x
        c = torch.concat([c, *p], dim=1)
        c = self.nlb(c)
        # c = self.seb(c)
        # c = self.rs3(c)
        p = [torch.concat([ps, c], dim=1) for ps in p]
        for index, m in enumerate(self.rs4):
            p[index] = m(p[index])
        return p

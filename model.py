import shlex

from torch import nn
import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import einops


# weight_initializer = nn.init.normal(std=2e-2)


class MlpBlock(nn.Module):
    def __init__(self, in_channels, mlp_dim, dropout_rate=0.0, use_bias=True):
        super(MlpBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_dim, bias=use_bias),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, in_channels, bias=use_bias),
        )

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=2e-2)

    def forward(self, x):
        out = self.mlp(x)
        return out


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


class UpSampleRatio(nn.Module):
    def __init__(self, in_channels, features, ratio, use_bias=True):
        super(UpSampleRatio, self).__init__()
        self.ratio = ratio
        self.Conv1x1 = nn.Conv2d(in_channels, features, (1, 1), bias=use_bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, c, h, w = x.shape
        x = Resize((int(h * self.ratio), int(w * self.ratio)), interpolation=InterpolationMode.BILINEAR)(x)
        out = self.Conv1x1(x).permute(0, 2, 3, 1)
        return out


class CALayer(nn.Module):
    """Squeeze-and-excitation block for channel attention.
    ref: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, features, reduction=4, use_bias=True):
        super(CALayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(features, features // reduction, (1, 1), bias=use_bias),
            nn.ReLU(),
            nn.Conv2d(features // reduction, features, (1, 1), bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        out = x * self.conv(x)
        return out


class RCAB(nn.Module):
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, features, reduction=4, lrelu_slope=0.2, use_bias=True):
        super(RCAB, self).__init__()
        self.ln = nn.LayerNorm(features)
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, (3, 3), bias=use_bias),
            nn.LeakyReLU(negative_slope=lrelu_slope),
            nn.Conv2d(features, features, (3, 3), bias=use_bias),
            CALayer(features, reduction, use_bias)
        )

    def forward(self, x):
        out = x + self.conv(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return out


class GridGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, in_channels, dim, use_bias=True):
        super(GridGatingUnit, self).__init__()
        self.in_channels = in_channels
        self.ln = nn.LayerNorm(in_channels // 2)
        self.dense = nn.Linear(dim, dim, bias=use_bias)
        nn.init.normal_(self.dense.weight, std=2e-2)

    def forward(self, x):
        u, v = torch.split(x, self.in_channels // 2, dim=-1)
        v = self.ln(v)
        v = torch.swapaxes(v, -1, -3)
        v = self.dense(v)
        v = torch.swapaxes(v, -1, -3)
        return u * (v + 1.0)


class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""

    def __init__(self, in_channels, gris_size, use_bias=True, factor=2, dropout_rate=0.0):
        super(GridGmlpLayer, self).__init__()
        self.grid_size = gris_size
        self.layers = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * factor, bias=use_bias),
            nn.GELU(),
            GridGatingUnit(in_channels * factor, gris_size[0]*gris_size[1], use_bias=use_bias),
            nn.Linear(in_channels * factor // 2, in_channels, bias=use_bias),
            nn.Dropout(dropout_rate)
        )

        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=2e-2)

    def forward(self, x):
        b, h, w, c = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x, patch_size=(fh, fw))
        x = x + self.layers(x)
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class BlockGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, in_channels, dim, use_bias=True):
        super(BlockGatingUnit, self).__init__()
        self.in_channels = in_channels
        self.ln = nn.LayerNorm(in_channels // 2)
        self.dense = nn.Linear(dim, dim, bias=use_bias)
        nn.init.normal_(self.dense.weight, std=2e-2)

    def forward(self, x):
        u, v = torch.split(x, self.in_channels // 2, dim=-1)
        v = self.ln(v)
        v = torch.swapaxes(v, -1, -2)
        v = self.dense(v)
        v = torch.swapaxes(v, -1, -2)
        return u * (v + 1.0)


class BlockGmlpLayer(nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""

    def __init__(self, in_channels, block_size, use_bias=True, factor=2, dropout_rate=0.0):
        super(BlockGmlpLayer, self).__init__()

        self.block_size = block_size
        self.layers = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * factor, bias=use_bias),
            nn.GELU(),
            BlockGatingUnit(in_channels * factor, block_size[0] * block_size[1], use_bias=use_bias),
            nn.Linear(in_channels * factor // 2, in_channels, bias=use_bias),
            nn.Dropout(dropout_rate)
        )

        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=2e-2)

    def forward(self, x):
        b, h, w, c = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        x = block_images_einops(x, patch_size=(fh, fw))
        x = x + self.layers(x)
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
    """The multi-axis gated MLP block."""

    def __init__(self, in_channels, block_size, grid_size, block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2,
                 use_bias=True, dropout_rate=0.0):
        super(ResidualSplitHeadMultiAxisGmlpLayer, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.block_size = block_size
        self.input_proj_factor = 2
        self.layers = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * input_proj_factor, bias=use_bias),
            nn.GELU()
        )
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=2e-2)

        self.gridgmlplayer = GridGmlpLayer(in_channels * input_proj_factor // 2, grid_size, use_bias,
                                           grid_gmlp_factor, dropout_rate)
        self.blockgmlplayer = BlockGmlpLayer(in_channels * input_proj_factor // 2, block_size, use_bias,
                                             block_gmlp_factor, dropout_rate)
        self.dense = nn.Sequential(
            nn.Linear(in_channels * input_proj_factor, in_channels, bias=use_bias),
            nn.Dropout(dropout_rate)
        )
        for module in self.dense.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=2e-2)

    def forward(self, x):
        shortcut = x
        x = self.layers(x)
        u, v = torch.split(x, self.in_channels * self.input_proj_factor // 2, dim=-1)
        u = self.gridgmlplayer(u)
        v = self.blockgmlplayer(v)
        x = torch.cat([u, v], dim=-1)
        x = self.dense(x)
        x = x + shortcut
        return x


class RDCAB(nn.Module):
    """Residual dense channel attention block. Used in Bottlenecks."""

    def __init__(self, in_channels, features, reduction=16, use_bias=True, dropout_rate=0.0):
        super(RDCAB, self).__init__()
        self.ln = nn.LayerNorm(in_channels)
        # (self, in_channels, mlp_dim, dropout_rate=0.0, use_bias=True)
        self.mlpblock = MlpBlock(in_channels, features, dropout_rate, use_bias)
        self.caylayer = CALayer(features, reduction, use_bias)

    def forward(self, x):
        y = self.ln(x)
        y = self.mlpblock(y)
        y = self.caylayer(y)
        x = x + y
        return x


class BottleneckBlock(nn.Module):
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""

    def __init__(self):
        super(BottleneckBlock, self).__init__()

    def forward(self):
        pass


class UNetEncoderBlock(nn.Module):
    """Encoder block in MAXIM."""

    def __init__(self):
        super(UNetEncoderBlock, self).__init__()

    def forward(self):
        pass


class UNetDecoderBlock(nn.Module):
    """Decoder block in MAXIM."""

    def __init__(self):
        super(UNetDecoderBlock, self).__init__()

    def forward(self):
        pass


class GetSpatialGatingWeights(nn.Module):
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, in_channels, block_size, grid_size, input_proj_factor=2, dropout_rate=0.0, use_bias=True):
        super(GetSpatialGatingWeights, self).__init__()
        self.grid_size = grid_size
        self.block_size = block_size
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * input_proj_factor, bias=use_bias),
            nn.GELU(),
        )
        self.dense_u = nn.Linear(grid_size[0]*grid_size[1], grid_size[0]*grid_size[1], bias=use_bias)
        nn.init.normal_(self.dense_u.weight, std=2e-2)
        nn.init.ones_(self.dense_u.bias)
        self.dense_v = nn.Linear(block_size[0]*block_size[1], block_size[0]*block_size[1], bias=use_bias)
        nn.init.normal_(self.dense_v.weight, std=2e-2)
        nn.init.ones_(self.dense_v.bias)
        self.dense = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=use_bias),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.layer1(x.permute(0, 2, 3, 1))
        u, v = torch.split(x, 2, dim=-1)

        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        u = block_images_einops(u, patch_size=(fh, fw))
        # dim_u = u.shape[-3]
        u = torch.swapaxes(u, -1, -3)
        u = self.dense_u(u)
        u = torch.swapaxes(u, -1, -3)
        u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))

        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        v = block_images_einops(v, patch_size=(fh, fw))
        # dim_v = v.shape[-2]
        v = torch.swapaxes(v, -1, -2)
        v = self.dense_v(v)
        v = torch.swapaxes(v, -1, -2)
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))

        x = torch.cat([u, v], dim=-1)
        x = self.dense(x).permute(0, 3, 1, 2)
        return x


class CrossGatingBlock(nn.Module):
    """Cross-gating MLP block."""

    def __init__(self, in_channels_x, in_channels_y, features, block_size, grid_size, dropout_rate=0.0, upsample_y=True,
                 use_bias=True):
        super(CrossGatingBlock, self).__init__()
        self.upsample_y = upsample_y
        self.ConvT_up = nn.ConvTranspose2d(in_channels_y, features, (2, 2), (2, 2), bias=use_bias)
        self.Conv1x1_x = nn.Conv2d(in_channels_x, features, (1, 1), bias=use_bias)
        self.Conv1x1_y = nn.Conv2d(in_channels_y, features, (1, 1), bias=use_bias)
        self.layer_x = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, features, bias=use_bias),
            nn.GELU()
        )
        self.GetSpatialGatingWeights_x = GetSpatialGatingWeights(features, block_size, grid_size,
                                                                 dropout_rate=dropout_rate, use_bias=use_bias)
        self.layer_y = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, features, bias=use_bias),
            nn.GELU()
        )
        self.GetSpatialGatingWeights_y = GetSpatialGatingWeights(features, block_size, grid_size,
                                                                 dropout_rate=dropout_rate, use_bias=use_bias)

        self.dense_x = nn.Sequential(
            nn.Linear(features, features, bias=use_bias),
            nn.Dropout(dropout_rate)
        )

        self.dense_y = nn.Sequential(
            nn.Linear(features, features, bias=use_bias),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, y):
        if self.upsample_y:
            y = self.ConvT_up(y)

        x = self.Conv1x1_x(x).permute(0, 2, 3, 1)
        y = self.Conv1x1_y(y).permute(0, 2, 3, 1)
        assert x.shape == y.shape

        shortcut_x = x
        shortcut_y = y
        x = self.layer_x(x)
        gx = self.GetSpatialGatingWeights_x(x)

        y = self.layer_y(y)
        gy = self.GetSpatialGatingWeights_y(y)

        y = y * gx
        y = shortcut_y + self.dense_y(y)

        x = x * gy
        x = shortcut_x + y + self.dense_x(x)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        return x, y


class SAM(nn.Module):
    """
    Supervised attention module for multi-stage training.
    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """

    def __init__(self, in_channels, features, out_channels, use_bias=True):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.Conv3x3_1 = nn.Conv2d(in_channels, features, (3, 3), bias=use_bias)
        self.Conv3x3_2 = nn.Conv2d(features, out_channels, (3, 3), bias=use_bias)
        self.Conv3x3_3 = nn.Sequential(
            nn.Conv2d(out_channels, features, (3, 3), bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x, x_image):
        x1 = self.Conv3x3_1(x)
        if self.out_channels == 3:
            image = self.Conv3x3_2(x) + x_image
        else:
            image = self.Conv3x3_2(x)
        x2 = self.Conv3x3_3(image)
        x1 = x1 * x2 + x
        return x1, image


class MAXIM(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 features: int = 64,
                 depth: int = 3,
                 num_stages: int = 2,
                 num_groups: int = 1,
                 use_bias: bool = True,
                 num_supervision_scales: int = 1,
                 lrelu_slope: float = 0.2,
                 use_global_mlp: bool = True,
                 use_cross_gating: bool = True,
                 high_res_stages: int = 2,
                 block_size_hr: tuple = (16, 16),
                 block_size_lr: tuple = (8, 8),
                 grid_size_hr: tuple = (16, 16),
                 grid_size_lr: tuple = (8, 8),
                 num_bottleneck_blocks: int = 1,
                 block_gmlp_factor: int = 2,
                 grid_gmlp_factor: int = 2,
                 input_proj_factor: int = 2,
                 channels_reduction: int = 4,
                 num_outputs: int = 3,
                 dropout_rate: float = 0.0):
        super(MAXIM, self).__init__()
        self.conv3x3_list = nn.ModuleList()
        for i in range(num_supervision_scales):
            self.conv3x3_list.append(nn.Conv2d(in_channels, features * (2 ** i), (3, 3), bias=use_bias))
            in_channels = features * (2 ** i)

        self.conv1x1_list = nn.ModuleList()
        for i in range(num_supervision_scales):
            self.conv3x3_list.append(nn.Conv2d(in_channels, features * (2 ** i), (1, 1), bias=use_bias))
            in_channels = features * (2 ** i)

        self.num_stages = num_stages
        self.num_supervision_scales = num_supervision_scales
        self.use_cross_gating = use_cross_gating
        self.block_size_hr = block_size_hr
        self.block_size_lr = block_size_lr
        self.grid_size_hr = grid_size_hr
        self.high_res_stages = high_res_stages

    def forward(self, x):
        b, c, h, w = x.shape
        shortcuts = [x]
        for i in range(1, self.num_supervision_scales):
            shortcuts.append(Resize((h // (2 ** i), w // (2 ** i)), interpolation=InterpolationMode.NEAREST)(x))

        outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []

        for idx_stage in range(self.num_stages):
            x_scales = []
            for i in range(self.num_supervision_scales):
                x_scale = self.conv3x3_list[i](shortcuts[i])

                if idx_stage > 0:
                    if self.use_cross_gating:
                        block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
                        grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr
                        x_scale, _ = CrossGatingBlock()
                    else:
                        x_scale = self.conv1x1_list[i](torch.cat([x_scale, sam_features.pop()], dim=1))
                x_scales.append(x_scale)

            encs = []
            x = x_scales[0]

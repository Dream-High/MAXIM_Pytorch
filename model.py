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
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        out = x * self.conv(y)
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
        out = x + self.conv(self.ln(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out


class GridGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, in_channels, dim, use_bias=True):
        super(GridGatingUnit, self).__init__()
        self.ln = nn.LayerNorm(in_channels // 2)
        self.dense = nn.Linear(dim, dim, bias=use_bias)
        nn.init.normal_(self.dense.weight, std=2e-2)

    def forward(self, x):
        b, h, w, c = x.shape
        u, v = torch.split(x, c // 2, dim=-1)
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
            GridGatingUnit(in_channels * factor, gris_size[0] * gris_size[1], use_bias=use_bias),
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
        self.calayer = CALayer(features, reduction, use_bias)

    def forward(self, x):
        y = self.ln(x)
        y = self.mlpblock(y)
        y = self.calayer(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + y
        return x


class BottleneckBlock(nn.Module):
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""

    def __init__(self, in_channels, features, block_size, grid_size, num_groups=1, block_gmlp_factor=2,
                 grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, dropout_rate=0.0, use_bias=True):
        super(BottleneckBlock, self).__init__()
        self.num_groups = num_groups
        self.Conv1x1 = nn.Conv2d(in_channels, features, (1, 1), bias=use_bias)
        self.layers = nn.Sequential(
            ResidualSplitHeadMultiAxisGmlpLayer(features, block_size, grid_size, block_gmlp_factor,
                                                grid_gmlp_factor, input_proj_factor, use_bias, dropout_rate),
            RDCAB(features, features, channels_reduction, use_bias)
        )

    def forward(self, x):
        x = self.Conv1x1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + self.layers(x)
        return x


class UNetEncoderBlock(nn.Module):
    """Encoder block in MAXIM."""

    def __init__(self, in_channels, features, block_size, grid_size, num_groups=1, lrelu_slope=0.2,
                 block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, dropout_rate=0.0,
                 downsample=True, use_global_mlp=True, use_bias=True, use_cross_gating=False):
        super(UNetEncoderBlock, self).__init__()
        self.num_groups = num_groups
        self.use_global_mlp = use_global_mlp
        self.use_cross_gating = use_cross_gating
        self.downsample = downsample
        self.Conv1x1 = nn.Conv2d(in_channels, features, (1, 1), bias=use_bias)
        self.RSHMAGL = ResidualSplitHeadMultiAxisGmlpLayer(features, block_size, grid_size, block_gmlp_factor,
                                                           grid_gmlp_factor, input_proj_factor, use_bias, dropout_rate)
        self.rcab = RCAB(features, channels_reduction, lrelu_slope, use_bias)
        self.cgb = CrossGatingBlock(in_channels, features, block_size, grid_size, dropout_rate, False,
                                    use_bias)
        self.Conv_down = nn.Conv2d(in_channels, features, (4, 4), (2, 2), bias=use_bias)

    def forward(self, x, skip=None, enc=None, dec=None, ):
        if skip is not None:
            x = torch.cat([x, skip], dim=-1)

        x = self.Conv1x1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        shortcut_long = x
        for i in range(self.num_groups):
            if self.use_global_mlp:
                x = self.RSHMAGL(x)

            x = self.rcab(x)

        x = x + shortcut_long

        if enc is not None and dec is not None:
            assert self.use_cross_gating
            x, _ = self.cgb(x, enc + dec)

        if self.downsample:
            x_down = self.Conv_down(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            return x_down, x
        else:
            return x


class UNetDecoderBlock(nn.Module):
    """Decoder block in MAXIM."""

    def __init__(self, in_channels, features, block_size, grid_size, num_groups=1, lrelu_slope=0.2,
                 block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, dropout_rate=0.0,
                 downsample=True, use_global_mlp=True, use_bias=True):
        super(UNetDecoderBlock, self).__init__()
        self.ConvT_up = nn.ConvTranspose2d(in_channels, features, (2, 2), (2, 2), bias=use_bias)
        self.encoderblock = UNetEncoderBlock(in_channels, features, block_size, grid_size, num_groups,
                                             lrelu_slope, block_gmlp_factor, grid_gmlp_factor, input_proj_factor,
                                             channels_reduction, dropout_rate, downsample, use_global_mlp, use_bias)

    def forward(self, x, bridge=None):
        x = self.ConvT_up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.encoder(x, skip=bridge)
        return x


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
        self.dense_u = nn.Linear(grid_size[0] * grid_size[1], grid_size[0] * grid_size[1], bias=use_bias)
        nn.init.normal_(self.dense_u.weight, std=2e-2)
        nn.init.ones_(self.dense_u.bias)
        self.dense_v = nn.Linear(block_size[0] * block_size[1], block_size[0] * block_size[1], bias=use_bias)
        nn.init.normal_(self.dense_v.weight, std=2e-2)
        nn.init.ones_(self.dense_v.bias)
        self.dense = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=use_bias),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        b, h, w, c = x.shape
        x = self.layer1(x)
        u, v = torch.split(x, c // 2, dim=-1)

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
        x = self.dense(x)
        return x


class CrossGatingBlock(nn.Module):
    """Cross-gating MLP block."""

    def __init__(self, in_channels, features, block_size, grid_size, dropout_rate=0.0, upsample_y=True,
                 use_bias=True):
        super(CrossGatingBlock, self).__init__()
        self.upsample_y = upsample_y
        self.ConvT_up = nn.ConvTranspose2d(in_channels, features, (2, 2), (2, 2), bias=use_bias)
        self.Conv1x1_x = nn.Conv2d(in_channels, features, (1, 1), bias=use_bias)
        self.Conv1x1_y = nn.Conv2d(in_channels, features, (1, 1), bias=use_bias)
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
            y = self.ConvT_up(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = self.Conv1x1_x(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.Conv1x1_y(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
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
        x = x.permute(0, 3, 1, 2)
        x_image = x_image.permute(0, 3, 1, 2)
        x1 = self.Conv3x3_1(x)
        if self.out_channels == 3:
            image = self.Conv3x3_2(x) + x_image
        else:
            image = self.Conv3x3_2(x)
        x2 = self.Conv3x3_3(image)
        x1 = x1 * x2 + x
        x1 = x1.permute(0, 2, 3, 1)
        image = image.permute(0, 2, 3, 1)
        return x1, image


class MAXIM_Single(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 features: int = 64,
                 depth: int = 3,
                 idx_stage: int = 0,
                 num_stages: int = 3,
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
        super(MAXIM_Single, self).__init__()
        self.idx_stage = idx_stage
        self.num_stages = num_stages
        self.num_supervision_scales = num_supervision_scales
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.use_cross_gating = use_cross_gating
        self.depth = depth

        self.stages_conv3x3_input = nn.ModuleList()
        self.stages_cgb_input = nn.ModuleList()
        self.stages_conv1x1_input = nn.ModuleList()

        self.stages_encoders = nn.ModuleList()
        self.stages_encoders_channels = []

        self.stages_bottleneckblocks = nn.ModuleList()

        self.stages_usr_cgb = nn.ModuleList()
        self.stages_cgb_cgb = nn.ModuleList()
        self.stages_conv1x1_cgb = nn.ModuleList()
        self.stages_conv3x3_cgb = nn.ModuleList()

        self.stages_usr_de = nn.ModuleList()
        self.stages_de = nn.ModuleList()
        self.stages_sam_de = nn.ModuleList()
        self.stages_conv3x3_de = nn.ModuleList()

        for idx_stage in range(self.num_stages):
            conv3x3_input = nn.ModuleList()
            cgb_input = nn.ModuleList()
            conv1x1_input = nn.ModuleList()
            t_channels = in_channels
            for i in range(num_supervision_scales):
                conv3x3_input.append(nn.Conv2d(t_channels, features * (2 ** i), (3, 3), bias=use_bias))
                if idx_stage > 0:
                    if self.use_cross_gating:
                        block_size = block_size_hr if i < high_res_stages else block_size_lr
                        grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                        cgb_input.append(CrossGatingBlock(t_channels, features * (2 ** i), block_size, grid_size,
                                                               dropout_rate, False, use_bias))
                    else:
                        conv1x1_input.append(nn.Conv2d(t_channels, features * (2 ** i), (1, 1), bias=use_bias))
                t_channels = features * (2 ** i)
            self.stages_conv3x3_input.append(conv3x3_input)
            self.stages_cgb_input.append(cgb_input)
            self.stages_conv1x1_input.append(conv1x1_input)

            encoders = nn.ModuleList()
            encoders_channels = []
            t_channels = in_channels
            for i in range(depth):
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                use_cross_gating_layer = True if idx_stage > 0 else False
                # self, in_channels, features, block_size, grid_size, num_groups = 1, lrelu_slope = 0.2,
                # block_gmlp_factor = 2, grid_gmlp_factor = 2, input_proj_factor = 2, channels_reduction = 4, dropout_rate = 0.0,
                # downsample = True, use_global_mlp = True, use_bias = True, use_cross_gating = False)
                encoders.append(
                    UNetEncoderBlock(t_channels, features * (2 ** i), block_size, grid_size, num_groups, lrelu_slope,
                                     block_gmlp_factor, grid_gmlp_factor, input_proj_factor, channels_reduction,
                                     dropout_rate, True, use_global_mlp, use_bias, use_cross_gating_layer)
                )
                t_channels = features * (2 ** i)
                encoders_channels.append(t_channels)
            self.stages_encoders.append(encoders)
            self.stages_encoders_channels.append(encoders_channels)

            bottleneckblocks = nn.ModuleList()
            for i in range(num_bottleneck_blocks):
                # (self, in_channels, features, block_size, grid_size, num_groups=1, block_gmlp_factor=2,
                # grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, dropout_rate=0.0, use_bias=True)
                bottleneckblocks.append(
                    BottleneckBlock(in_channels, (2 ** (self.depth - 1)) * self.features, block_size_lr, block_size_lr,
                                    num_groups, block_gmlp_factor, grid_gmlp_factor, input_proj_factor, channels_reduction,
                                    dropout_rate, use_bias)
                )
            self.stages_bottleneckblocks.append(bottleneckblocks)

            usr_cgb = nn.ModuleList()
            cgb_cgb = nn.ModuleList()
            conv1x1_cgb = nn.ModuleList()
            conv3x3_cgb = nn.ModuleList()
            t_channels = in_channels
            for i in reversed(range(depth)):
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                self.usr_cgb.append(nn.ModuleList())
                for j in range(depth):
                    self.usr_cgb[i].append(
                        UpSampleRatio(self.encoders_channels[j], (2 ** i) * features, 2 ** (j - i), use_bias))
                if self.use_cross_gating:
                    # (self, in_channels, features, block_size, grid_size, dropout_rate=0.0, upsample_y=True,
                    # use_bias=True)
                    self.cgb_cgb.append(
                        CrossGatingBlock(in_channels, (2 ** i) * features, block_size, grid_size, dropout_rate, True,
                                         use_bias)
                    )
                else:
                    self.conv1x1_cgb.append(nn.Conv2d(t_channels, features * (2 ** i), (1, 1), bias=use_bias))
                    self.conv3x3_cgb.append(nn.Conv2d(t_channels, features * (2 ** i), (3, 3), bias=use_bias))
            self.stages_usr_cgb.append(usr_cgb)
            self.stages_cgb_cgb.append(cgb_cgb)
            self.stages_conv1x1_cgb.append(conv1x1_cgb)
            self.stages_conv3x3_cgb.append(conv3x3_cgb)

            usr_de = nn.ModuleList()
            de = nn.ModuleList()
            sam_de = nn.ModuleList()
            conv3x3_de = nn.ModuleList()
            t_channels = in_channels
            for i in reversed((range(depth))):
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                self.usr_de.append(nn.ModuleList())
                for j in range(depth):
                    self.usr_de[i].append(
                        UpSampleRatio(self.encoders_channels[j], (2 ** i) * features, 2 ** (depth - j - 1 - i), use_bias)
                    )
                # (self, in_channels, features, block_size, grid_size, num_groups=1, lrelu_slope=0.2,
                # block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, dropout_rate=0.0,
                # downsample=True, use_global_mlp=True, use_bias=True)
                self.de.append(
                    UNetDecoderBlock(in_channels, (2 ** i) * features, block_size, grid_size, num_groups, lrelu_slope,
                                     block_gmlp_factor, grid_gmlp_factor, input_proj_factor, channels_reduction,
                                     dropout_rate, True, use_global_mlp, use_bias)
                )
                if i < self.num_supervision_scales:
                    if idx_stage < self.num_satges - 1:
                        self.sam_de.append(SAM(t_channels, (2 ** i) * features, num_outputs, use_bias))
                    else:
                        self.conv3x3_de.append(nn.Conv2d(t_channels, num_outputs, (3, 3), bias=use_bias))
            self.stages_usr_de.append(usr_de)
            self.stages_de.append(de)
            self.stages_sam_de.append(sam_de)
            self.stages_conv3x3_de.append(conv3x3_de)

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
                        x_scale, _ = self.cgb_list[i](x_scale, sam_features.pop())
                    else:
                        x_scale = self.conv1x1_list[i](torch.cat([x_scale, sam_features.pop()], dim=1))
                x_scales.append(x_scale)

            encs = []
            x = x_scales[0]

            for i in range(self.depth):
                pass

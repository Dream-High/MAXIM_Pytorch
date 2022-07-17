from torch import nn
import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import einops

# weight_initializer = nn.init.normal(std=2e-2)


class MlpBlock(nn.Module):
    def __init__(self, in_channels, mlp_channels, dropout_rate=0.0, use_bias=True):
        super(MlpBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_channels, bias=use_bias),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_channels, in_channels, bias=use_bias),
        )

    def forward(self, x):
        ''' x: [batch_size * channel * height * width]'''
        # out = self.mlp(x.transpose(1, 3).transpose(1, 2)).transpose(1, 3).transpose(2, 3)
        out = self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, channels, height, width = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "b c (gh fh) (gw fw) -> b c (gh gw) (fh fw)",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "b c (gh gw) (fh fw) -> b c (gh fh) (gw fw)",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


class UpSampleRatio(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, use_bias=True):
        super(UpSampleRatio, self).__init__()
        self.ratio = ratio
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1), bias=use_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = Resize((int(h * self.ratio), int(w * self.ratio)), interpolation=InterpolationMode.BILINEAR)(x)
        out = self.conv(x)
        return out


class CALayer(nn.Module):
    """Squeeze-and-excitation block for channel attention.
    ref: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels, out_channels, reduction=4, use_bias=True):
        super(CALayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, (1, 1), bias=use_bias),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction, out_channels, (1, 1), bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        out = x * self.conv(x)
        return out


class RCAB(nn.Module):
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, in_channels, out_channels, dim=1, reduction=4, lrelu_slope=0.2, use_bias=True):
        super(RCAB, self).__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(in_channels, out_channels, (3, 3), bias=use_bias),
            nn.LeakyReLU(negative_slope=lrelu_slope),
            nn.Conv2d(out_channels, out_channels, (3, 3), bias=use_bias),
            CALayer(out_channels, out_channels, reduction, use_bias)
        )

    def forward(self, x):
        out = x + self.conv(x)
        return out


class GridGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, dim=1, use_bias=True):
        super(GridGatingUnit, self).__init__()
        # self.dim = dim
        # self.use_bias = use_bias
        self.layernorm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim, use_bias)
        # self.layers = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim, use_bias)
        # )
        pass

    def forward(self, x):
        u, v = torch.split(x, 2, dim=1)
        v = self.layernorm(v)
        pass


class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self, in_channels, gris_size, dim=1, use_bias=True, factor=2, dropout_rate=0.0):
        self.grid_size = gris_size
        self.laynorm = nn.LayerNorm(dim)
        super(GridGmlpLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, in_channels * factor, bias=use_bias),
            nn.GELU(),
            GridGatingUnit(dim, use_bias=use_bias),
            nn.Linear(in_channels * factor, in_channels, bias=use_bias),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x, patch_size=(fh, fw))
        x = x.permute(1, 2, 3, 0)


class BlockGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self):
        super(BlockGatingUnit, self).__init__()
        pass
    
    def forward(self, x):
        pass


class BlockGmlpLayer(nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self):
        super(BlockGmlpLayer, self).__init__()
        pass
    
    def forward(self, x):
        pass


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self):
        super(ResidualSplitHeadMultiAxisGmlpLayer, self).__init__()
    
    def forward(self):
        pass


class RDCAB(nn.Module):
    """Residual dense channel attention block. Used in Bottlenecks."""
    def __init__(self):
        super(RDCAB, self).__init__()
    
    def forward(self):
        pass


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
    def __init__(self):
        super(GetSpatialGatingWeights, self).__init__()
    
    def forward(self):
        pass


class CrossGatingBlock(nn.Module):
    """Cross-gating MLP block."""
    def __init__(self, in_channels, features, block_size, grid_size, dropout_rate=0.0, upsample_y=True, use_bias=True):
        super(CrossGatingBlock, self).__init__()
        self.upsample_y = upsample_y
        self.ConvT_up = nn.ConvTranspose2d(in_channels, features, (2, 2), (2, 2), bias=use_bias)
        self.Conv1x1_x = nn.Conv2d(in_channels, features, (1, 1), bias=use_bias)
        self.Conv1x1_y = nn.Conv2d(in_channels, features, (1, 1), bias=use_bias)
    
    def forward(self, x, y):
        if self.upsample_y:
            y = self.ConvT_up(y)
        
        x = self.Conv1x1_x(x)
        y = self.Conv1x1_y(y)
        assert x.shape == y.shape
        b, c, h, w = x.shape
        shortcut_x = x
        shortcut_y = y
        
        


class SAM(nn.Module):
    """
    Supervised attention module for multi-stage training.
    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """
    def __init__(self):
        super(SAM, self).__init__()
    
    def forward(self):
        pass


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
            self.conv3x3_list.append(nn.Conv2d(in_channels, features*(2**i), (3, 3), bias=use_bias))
            in_channels = features*(2**i)
        
        self.conv1x1_list = nn.ModuleList()
        for i in range(num_supervision_scales):
            self.conv3x3_list.append(nn.Conv2d(in_channels, features*(2**i), (1, 1), bias=use_bias))
            in_channels = features*(2**i)
        
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
            shortcuts.append(Resize((h // (2**i), w // (2**i)), interpolation=InterpolationMode.NEAREST)(x))
        
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
            

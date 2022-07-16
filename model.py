from torch import nn
import torch
from torchvision.transforms import Resize
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
        x = Resize((int(h * self.ratio), int(w * self.ratio)))(x)
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

    def __init__(self, in_channels, out_channels, dim, reduction=4, lrelu_slope=0.2, use_bias=True):
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


# class GridGatingUnit(nn.Module):
#     """A SpatialGatingUnit as defined in the gMLP paper.
#     The 'spatial' dim is defined as the second last.
#     If applied on other dims, you should swapaxes first.
#     """
#     def __init__(self, dim, use_bias=True):
#         super(GridGatingUnit, self).__init__()
#         # self.dim = dim
#         # self.use_bias = use_bias
#         self.layernorm = nn.LayerNorm(dim)
#         self.linear = nn.Linear(dim, dim, use_bias)
#         # self.layers = nn.Sequential(
#         #     nn.LayerNorm(dim),
#         #     nn.Linear(dim, dim, use_bias)
#         # )
#
#     def forward(self, x):
#         u, v = torch.split(x, 2, dim=-1)
#         v = self.layernorm(v)
#
#
#
#     @nn.compact
#     def __call__(self, x):
#         u, v = jnp.split(x, 2, axis=-1)
#         v = nn.LayerNorm(name="intermediate_layernorm")(v)
#         n = x.shape[-3]  # get spatial dim
#         v = jnp.swapaxes(v, -1, -3)
#         v = nn.Dense(n, use_bias=self.use_bias, kernel_init=weight_initializer)(v)
#         v = jnp.swapaxes(v, -1, -3)
#         return u * (v + 1.)


class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self, gris_size, dim, use_bias=True, factor=2, dropout_rate=0.0):
        super(GridGmlpLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear()
        )


  grid_size: Sequence[int]
  use_bias: bool = True
  factor: int = 2
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, num_channels = x.shape
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    x = block_images_einops(x, patch_size=(fh, fw))
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    y = nn.LayerNorm(name="LayerNorm")(x)
    y = nn.Dense(num_channels * self.factor, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="in_project")(y)
    y = nn.gelu(y)
    y = GridGatingUnit(use_bias=self.use_bias, name="GridGatingUnit")(y)
    y = nn.Dense(num_channels, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="out_project")(y)
    y = nn.Dropout(self.dropout_rate)(y, deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    return x
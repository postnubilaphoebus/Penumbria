import torch
import torch.nn as nn
from einops import rearrange
#from monai.networks.blocks import PatchEmbeddingBlock
import einops
import torch.nn.functional as F
from .VisionLSTM import *
import sys
from .frn import FilterResponseNorm3d
import os
import math
import torch.fft

#SUPPORTED_EMBEDDING_TYPES = ("conv", "perceptron")
# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from einops.layers.torch import Rearrange


def trunc_normal_(tensor, mean=0.0, std=0.02, a=-2.0, b=2.0):
    # minimal replacement for timm-style trunc_normal_
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp > a) & (tmp < b)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)
    return tensor


class PatchEmbeddingBlock(nn.Module):
    """
    Patch embedding with optional convolutional or perceptron-style patchification
    and learnable positional embeddings.
    In the style of "Dosovitskiy, A. (2020). 
    An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929."

    Input shape:  (B, C, D, H, W) for spatial_dims=3
    Output shape: (B, N_patches, hidden_size)
    """

    def __init__(
        self,
        in_channels: int,
        img_size,
        patch_size,
        hidden_size: int,
        num_heads: int,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        if pos_embed not in {"conv", "perceptron"}:
            raise ValueError("pos_embed must be 'conv' or 'perceptron'")

        self.pos_embed_type = pos_embed
        self.spatial_dims = spatial_dims

        if isinstance(img_size, int):
            img_size = (img_size,) * spatial_dims
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * spatial_dims

        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size must be <= img_size")
            if pos_embed == "perceptron" and m % p != 0:
                raise ValueError(
                    "For perceptron embedding, img_size must be divisible by patch_size"
                )

        self.n_patches = math.prod(m // p for m, p in zip(img_size, patch_size))
        self.patch_dim = in_channels * math.prod(patch_size)

        # --- patch embedding ---
        if pos_embed == "conv":
            if spatial_dims == 2:
                self.patch_embeddings = nn.Conv2d(
                    in_channels,
                    hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
            elif spatial_dims == 3:
                self.patch_embeddings = nn.Conv3d(
                    in_channels,
                    hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
            else:
                raise ValueError("Only spatial_dims=2 or 3 supported")
        else:
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = (
                f"b ({' '.join([c[0] for c in chars])}) "
                f"({' '.join([c[1] for c in chars])} c)"
            )
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}

            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len),
                nn.Linear(self.patch_dim, hidden_size),
            )

        # --- positional embedding ---
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.n_patches, hidden_size)
        )

        self.dropout = nn.Dropout(dropout_rate)

        trunc_normal_(self.position_embeddings, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embeddings(x)

        if self.pos_embed_type == "conv":
            x = x.flatten(2).transpose(1, 2)

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class UniformSpectralDropout3d(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, X_freq, shape=None):
        dropout_mask = (torch.rand_like(X_freq.real) >= self.p).to(X_freq.dtype)
        dropout_mask[..., 0, 0, 0] = 1.0
        return X_freq * dropout_mask

class GlobalZernikeConv3d(nn.Module):
    def __init__(self, j_indices=[4, 12], dropout_p=0.2):
        super().__init__()
        self.j_indices = j_indices
        self.dropout_p = dropout_p
        self.spectral_dropout = UniformSpectralDropout3d(p = dropout_p)
        self.alphas = nn.Parameter(torch.zeros(len(j_indices)))
        
    def forward(self, x):
        # 1. Real FFT to frequency domain
        # X_freq shape: [B, C, D, H, W//2 + 1]
        X_freq = torch.fft.rfftn(x, dim=(-3, -2, -1))
        
        # 2. Apply Spectral Dropout (Training only)
        if self.training and self.dropout_p > 0 and torch.rand(1) > 0.5:
            X_freq = self.spectral_dropout(X_freq, x.shape)
        
        # 3. Generate and apply Phase Mask
        mask = self._get_rfft_zernike_mask(x.shape[2:], x.device, x.dtype)
        weight = torch.exp(1j * mask)
        X_aberrated = X_freq * weight
        
        # 4. Inverse Real FFT
        return torch.fft.irfftn(X_aberrated, s=x.shape[2:], dim=(-3, -2, -1))

    def _get_rfft_zernike_mask(self, shape, device, dtype):
        D, H, W = shape
        z_freq = torch.fft.fftfreq(D, device=device)
        y_freq = torch.fft.fftfreq(H, device=device)
        x_freq = torch.fft.rfftfreq(W, device=device)
        
        Z, Y, X = torch.meshgrid(z_freq, y_freq, x_freq, indexing='ij')
        R = torch.sqrt(X**2 + Y**2 + Z**2)

        constrained_alphas = torch.tanh(self.alphas) * 2.0

        total_phase = torch.zeros_like(X)
        for idx, j in enumerate(self.j_indices):
            F = self._get_zernike_mode(j, X, Y, Z, R)
            total_phase += constrained_alphas[idx] * F
            
        return total_phase

    def _get_zernike_mode(self, j, X, Y, Z, R):
        if j == 0: return torch.ones_like(X)
        if j == 3: return Z 
        if j == 4: return 2*R**2 - 1 
        if j == 12: return 6.*R**4 - 6.*R**2 + 1. 
        return torch.zeros_like(X)

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            FilterResponseNorm3d(out_channels)#nn.InstanceNorm3d(out_channels, affine = True)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = FilterResponseNorm3d(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)

        self.norm2 = FilterResponseNorm3d(width)
        self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = FilterResponseNorm3d(out_channels)


    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down

        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels,
                 depth=24,
                 dim=1024,
                 drop_path_rate=0.0,
                 stride=None,
                 alternation="bidirectional",
                 drop_path_decay=False,
                 legacy_norm=False):
        super().__init__()

        self.norm_type = "instance"
        self.activation = "mish"
        self.zernike = GlobalZernikeConv3d(j_indices = [3, 4, 12])
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.norm1 = FilterResponseNorm3d(out_channels)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)
        self.patch_embed = PatchEmbeddingBlock(in_channels=out_channels * 8,
                                               img_size=img_dim // 16,
                                               patch_size=2,
                                               hidden_size=256,
                                               num_heads=1,
                                               spatial_dims=3)
        self.conv2 = nn.Conv3d(out_channels * 8, 512,
                               kernel_size=3, stride=1, padding=1)
        
        if self.norm_type == "instance":
            self.norm2 = FilterResponseNorm3d(512)
        else:
            self.norm2 = nn.BatchNorm3d(512)
        self.alternation = alternation
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # directions
        directions = []
        if alternation == "bidirectional":
            for i in range(depth):
                if i % 2 == 0:
                    directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
                else:
                    directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        else:
            raise NotImplementedError(f"invalid alternation '{alternation}'")

        # blocks
        self.blocks = nn.ModuleList(
            [
                ViLBlock(
                    dim=dim,
                    drop_path=dpr[i],
                    direction=directions[i],
                )
                for i in range(depth)
            ]
        )
        if legacy_norm:
            self.legacy_norm = LayerNorm(dim, bias=False)
        else:
            self.legacy_norm = nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.output_shape = ((img_dim // 16) // 2, dim)

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}

    def forward(self, x):
        x = self.zernike(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = x

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.patch_embed(x)
        x = einops.rearrange(x, "b ... d -> b (...) d")
        
        for block in self.blocks:
            x = block(x)
    
        x = self.legacy_norm(x)
        x = self.norm(x)
        x = rearrange(x, "b (x y z) c -> b c x y z", x=self.output_shape[0], y=self.output_shape[0],
                      z=self.output_shape[0])
        return x, x1, x2, x3


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=scale_factor * 2, mode='trilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            FilterResponseNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            FilterResponseNorm3d(out_channels)
        )


    def forward(self, x, x_concat=None):
        if x.shape[2] == 3:
            x = self.upsample1(x)
        else:
            x = self.upsample(x)
        if x_concat is not None:
            diffD = x_concat.size()[2] - x.size()[2]  # Depth
            diffY = x_concat.size()[3] - x.size()[3]  # Height
            diffX = x_concat.size()[4] - x.size()[4]  # Width

            # Apply padding on each side of the last three dimensions
            x = F.pad(x, [
                diffX // 2, diffX - diffX // 2,  # Padding for Width
                diffY // 2, diffY - diffY // 2,  # Padding for Height
                diffD // 2, diffD - diffD // 2   # Padding for Depth
            ])
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x



class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))
        self.conv1 = nn.Conv3d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        return x

class UVixLSTM(nn.Module):
    def __init__(self, class_num, 
                     img_dim=96,
                     in_channels=1,
                     out_channels=64,
                     depth=12,
                     dim=256):
        super().__init__()
        self.encoder = Encoder(img_dim, in_channels, out_channels,
                                   depth, dim)
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x_main = self.decoder(x, x1, x2, x3)
        return x_main, None
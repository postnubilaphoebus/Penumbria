''' This file adapts code from U-VixLSTM by Pallabi Dutta, Soham Bose, Swalpa Kumar Roy, and Sushmita Mitra 
URL: https://github.com/duttapallabi2907/U-VixLSTM. Paper: https://arxiv.org/abs/2406.16993'''
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

        self.msg_layer = MultiScaleGraphVolume(volume_side=64, num_levels=4, in_channels=in_channels, out_channels=256,feat_channels=16)

        self.norm1 = FilterResponseNorm3d(out_channels)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)
        self.patch_embed = PatchEmbeddingBlock(in_channels=out_channels * 8,
                                               img_size=img_dim // 16,
                                               patch_size=2,
                                               hidden_size=256,
                                               num_heads=1,
                                               pos_embed = "perceptron",
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
                    dim=512,
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
        self.norm = nn.LayerNorm(512, eps=1e-6)

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

    def drop_branches(self, msg_feat, unet_feat, p_drop=0.15, training=True):
        if not training:
            return msg_feat, unet_feat
        r = torch.rand(1).item()
        if r < p_drop:
            return torch.zeros_like(msg_feat), unet_feat
        elif r < 2 * p_drop:
            return msg_feat, torch.zeros_like(unet_feat)
        return msg_feat, unet_feat

    def forward(self, x, prompt = None):
        x = self.zernike(x)
        msg = self.msg_layer(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = x

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        x = self.patch_embed(x)
        x = einops.rearrange(x, "b ... d -> b (...) d")

        msg = msg.reshape(x.shape)
        x, msg = self.drop_branches(msg, x, p_drop = 0.15, training=self.training)
        x = torch.cat([x, msg], dim = -1)
        if prompt is not None:
            prompt = prompt.reshape(x.shape)
            x = x + prompt
        
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

class FRNConvDownsample3D_3Steps(nn.Module):
    def __init__(self, in_channels=1, mid_channels=64, out_channels=256):
        super().__init__()
        
        # Step 1: Dilated conv (~7 effective) + FRN + MaxPool (stride 4)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.frn1 = FilterResponseNorm3d(mid_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=4, stride=4)
        
        # Step 2: Smaller conv + FRN + MaxPool (stride 2)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.frn2 = FilterResponseNorm3d(mid_channels)
        self.pool2 = nn.MaxPool3d(kernel_size=4, stride=4)
        
        # Step 3: Final conv to out_channels + FRN + MaxPool (stride 2)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.frn3 = FilterResponseNorm3d(out_channels)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x); x = self.frn1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.frn2(x); x = self.pool2(x)
        x = self.conv3(x); x = self.frn3(x); x = self.pool3(x)
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
        self.cell_clue_layer = FRNConvDownsample3D_3Steps(in_channels=1, mid_channels=64, out_channels=512)

    def forward(self, x, prompt = None):
        if prompt is not None:
            prompt = self.cell_clue_layer(prompt)
        x, x1, x2, x3 = self.encoder(x, prompt)
        x_main = self.decoder(x, x1, x2, x3)
        return x_main, None


import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def squash(s: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Capsule squash non-linearity.

    Maps vector ``s`` so its ℓ₂-norm lies in (0, 1):
        v = (‖s‖² / (1 + ‖s‖²)) · (s / ‖s‖)

    Short vectors → near-zero output  (existence unlikely).
    Long  vectors → near-unit output  (existence likely).

    Args:
        s:   Tensor of shape (..., C, ...) where ``dim`` indexes the capsule dim.
        dim: Dimension along which to compute the norm.
    """
    norm_sq = (s * s).sum(dim=dim, keepdim=True)
    norm    = norm_sq.sqrt()
    return (norm_sq / (1.0 + norm_sq)) * (s / (norm + eps))


def _make_cascade_proj(feat_channels: int, log2_ratio: int) -> nn.Module:
    """
    Cascade of ``log2_ratio`` stride-2 Conv3d + GroupNorm + GELU blocks.

    Used to spatially reduce a feature map by exactly 2^log2_ratio without
    the artefacts that large-stride single convolutions can introduce.
    """
    if log2_ratio == 0:
        return nn.Identity()
    g = min(8, feat_channels)
    layers: List[nn.Module] = []
    for _ in range(log2_ratio):
        layers += [
            nn.Conv3d(feat_channels, feat_channels, 3,
                      stride=2, padding=1, bias=False),
            nn.GroupNorm(g, feat_channels),
            nn.GELU(),
        ]
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Scale-level feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class ScaleLevelExtractor(nn.Module):
    """
    Feature extraction at one grid-scale level.

    Flow:
        input (B, C_in, N, N, N)
          │
          ├─ Conv3d(dilation=d)  ← receptive field: (2d+1)³
          ├─ GroupNorm + GELU
          ├─ Conv3d(1×1×1)       ← channel mixing
          ├─ GroupNorm + GELU
          └─ AvgPool3d(stride=s) ← sample the grid
          │
        output (B, C, N//s, N//s, N//s)

    Args:
        in_channels:   Input channels C_in.
        feat_channels: Output channels C.
        dilation:      Conv dilation.  Set to 2^k for level k.
        stride:        Grid sampling stride.  Set to 2^k for level k.
    """

    def __init__(
        self,
        in_channels:   int,
        feat_channels: int,
        dilation:      int,
        stride:        int,
    ) -> None:
        super().__init__()
        g = min(8, feat_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels,
                      kernel_size=3, dilation=dilation,
                      padding=dilation, bias=False),
            nn.GroupNorm(g, feat_channels),
            nn.GELU(),
            nn.Conv3d(feat_channels, feat_channels,
                      kernel_size=1, bias=False),
            nn.GroupNorm(g, feat_channels),
            nn.GELU(),
        )
        self.pool = nn.AvgPool3d(stride, stride) if stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# Spatial capsule router
# ─────────────────────────────────────────────────────────────────────────────

class SpatialCapsuleRouter(nn.Module):
    """
    Routes capsule features from a fine grid level to the adjacent coarser one.

    Key idea — learned coupling prior
    ──────────────────────────────────
    ``vote_conv`` is a stride-2, kernel=(hood × hood × hood) Conv3d that produces
    ``hood³`` vote vectors per coarse position from the fine feature map.
    The vote vectors answer: "given what I (fine-level position i) see, here is
    my prediction for what coarse-level position j should encode."

    ``log_prior``  (shape [1, hood³, 1, 1, 1], init = 0)
        Learnable log-prior routing logits — one scalar per neighbourhood
        offset.  The softmax gives the base coupling probability for each
        offset direction before agreement refinement.

        • At init   → uniform (1/hood³ for each offset)
        • After training → some offsets amplified, others suppressed
          → potentially sparse / small-world structure

    Dynamic routing (Sabour et al., 2017)
    ──────────────────────────────────────
    Inside each forward call, ``num_routing_iters`` agreement steps refine
    the coupling temporarily (these runtime updates do not back-propagate
    through themselves; only the learned ``log_prior`` accumulates gradient).

        for i in range(num_routing_iters):
            c  = softmax(b)                  # coupling coefficients
            s  = Σ_k c_k · votes_k + coarse  # aggregated capsule (residual)
            v  = squash(s)                   # capsule output
            b += votes · v                   # agreement update

    Args:
        feat_channels:     Feature / capsule dimension C.
        neighborhood:      Side length (hood) of the routing window.
                           Neighbourhood covers hood³ fine-grid positions.
        num_routing_iters: Inner routing iterations (3 recommended).
    """

    def __init__(
        self,
        feat_channels:     int,
        neighborhood:      int = 3,
        num_routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.C     = feat_channels
        self.hood  = neighborhood
        self.K3    = neighborhood ** 3
        self.iters = num_routing_iters
        g = min(8, feat_channels)

        # Produces hood³ vote vectors per coarse position.
        # stride=2 halves the spatial dims (fine → coarse).
        self.vote_conv = nn.Conv3d(
            feat_channels,
            feat_channels * self.K3,
            kernel_size=neighborhood,
            stride=2,
            padding=neighborhood // 2,
            bias=False,
        )

        # ─ The key "connectedness" parameter ─────────────────────────────
        # One log-prior per neighbourhood offset.
        # Initialized to 0 → uniform softmax → uniform routing.
        # SGD pushes these toward a learned topology.
        self.log_prior = nn.Parameter(torch.zeros(1, self.K3, 1, 1, 1))

        self.out_norm = nn.GroupNorm(g, feat_channels)

        # self.rnn = nn.GRU(self.K3, self.K3 // 2, 1, bidirectional = True)
        # self.linear_add = nn.Linear(self.K3-1, self.K3)

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fine:   (B, C, 2n, 2n, 2n)  finer level features.
            coarse: (B, C,  n,  n,  n)  coarser level features (used as residual
                                         and capsule initialisation).
        Returns:
            Updated coarser capsules, shape (B, C, n, n, n).
        """
        B, C = fine.shape[:2]

        # ── Vote projection ──────────────────────────────────────────────
        # votes_flat: (B, C·K³, n, n, n)
        votes_flat = self.vote_conv(fine)
        n = votes_flat.shape[2]          # coarse spatial size

        # votes: (B, K³, C, n, n, n)
        votes = votes_flat.view(B, self.K3, C, n, n, n)

        # ── Routing logit initialisation ─────────────────────────────────
        # Clone so that in-loop additions don't affect grad of log_prior.
        # Shape: (B, K³, n, n, n)
        b = self.log_prior.expand(B, self.K3, n, n, n).clone()

        # ── Dynamic routing iterations ───────────────────────────────────
        v: torch.Tensor = coarse
        for i in range(self.iters):

            # Coupling coefficients: softmax over K³ neighbourhood offsets
            c = F.softmax(b, dim=1)                             # (B, K³, n,n,n)

            # Weighted sum of votes + residual from coarse-level features
            #   c.unsqueeze(2): (B, K³, 1, n, n, n)
            #   votes:          (B, K³, C, n, n, n)
            s = (c.unsqueeze(2) * votes).sum(dim=1) + coarse   # (B, C, n,n,n)

            # Squash: capsule output
            v = squash(s, dim=1)

            # Agreement update (skip on final iteration — b not used again)
            if i < self.iters - 1:
                # Dot product votes · v, summed over C → (B, K³, n,n,n)
                agreement = (votes * v.unsqueeze(1)).sum(dim=2)
                b = b + agreement

        return self.out_norm(v)

    # ─────────────────────────────────────────────────────────────────────

    def routing_weights(self) -> torch.Tensor:
        """
        Inspect the learned coupling prior: softmax(log_prior).

        Returns:
            Tensor of shape (hood³,) — probability mass over neighbourhood
            offsets.  Uniform (1/hood³) at init; diverges during training.
        """
        with torch.no_grad():
            return F.softmax(self.log_prior.view(-1), dim=0)

    def routing_entropy(self) -> float:
        """
        Shannon entropy of the current routing distribution (in bits).
        Max = log₂(hood³); converges toward 0 for highly peaked distributions.
        """
        p = self.routing_weights()
        return float(-(p * (p + 1e-12).log2()).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Main layer
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleGraphVolume(nn.Module):
    """
    Multi-scale graph-sampling layer for 3-D cubic volumes.

    Combines:
      • Regular-grid sampling at K coarseness levels (stride 2^k per level).
      • Dilated 3-D convolutions: dilation 2^k ∝ coarseness level.
      • Capsule-style dynamic routing between adjacent levels, with
        **learned coupling priors** that start uniform and diverge toward
        sparse / small-world topology.
      • Final multi-level fusion at the coarsest spatial resolution.

    Grid layout (level 0 = finest, level K-1 = coarsest)
    ─────────────────────────────────────────────────────────────────────
    Level   Stride   Dilation   Grid pts/dim   Routing
      0       2⁰=1    2⁰=1         N/1          ─── router 0 ───►
      1       2¹=2    2¹=2         N/2          ─── router 1 ───►
      2       2²=4    2²=4         N/4          ─── router 2 ───►
      3       2³=8    2³=8         N/8

    After routing, all levels are projected to grid size N/2^(K-1) and fused.

    Args:
        volume_side (int):
            Side length N of the cubic input volume.
            Must satisfy: ``volume_side % 2^(num_levels-1) == 0``.
        num_levels (int):
            Number of grid coarseness levels K.  Default 4.
        in_channels (int):
            Input feature channels.
        feat_channels (int):
            Internal feature channels at each level.
        out_channels (int | None):
            Output channels.  Defaults to ``feat_channels``.
        num_routing_iters (int):
            Capsule routing iterations per SpatialCapsuleRouter.  Default 3.
        neighborhood (int):
            Side length of the routing window in fine-grid coordinates.
            Larger → more expressive routing, more parameters.  Default 3.

    Shape:
        - Input:  ``(B, in_channels, N, N, N)``
        - Output: ``(B, out_channels, N//2^(K-1), N//2^(K-1), N//2^(K-1))``

    Example::

        >>> layer = MultiScaleGraphVolume(volume_side=32, num_levels=4,
        ...                               in_channels=1, feat_channels=16)
        >>> x = torch.randn(2, 1, 32, 32, 32)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 16, 4, 4, 4])

        >>> # inspect learned routing topology after some training
        >>> for k, probs in enumerate(layer.routing_topology()):
        ...     print(f"Router {k}→{k+1}: entropy = {layer.routers[k].routing_entropy():.3f} bits")
    """

    def __init__(
        self,
        volume_side:       int,
        num_levels:        int = 4,
        in_channels:       int = 1,
        feat_channels:     int = 32,
        out_channels:      Optional[int] = None,
        num_routing_iters: int = 3,
        neighborhood:      int = 3,
    ) -> None:
        super().__init__()

        self.N    = volume_side
        self.K    = num_levels
        self.C    = feat_channels
        self.Cout = out_channels or feat_channels

        # ── Validation ───────────────────────────────────────────────────────
        min_divisor = 2 ** (num_levels - 1)
        if volume_side % min_divisor != 0:
            raise ValueError(
                f"volume_side={volume_side} must be divisible by "
                f"2^(num_levels-1)={min_divisor} so all grid levels are integers."
            )

        # ── Grid parameters per level ─────────────────────────────────────────
        # stride_k = 2^k,  dilation_k = 2^k
        self.strides    = [2 ** k for k in range(num_levels)]   # 1, 2, 4, 8, …
        self.dilations  = [2 ** k for k in range(num_levels)]   # 1, 2, 4, 8, …
        self.grid_sizes = [volume_side // s for s in self.strides]

        # ── Feature extractors: one per level ─────────────────────────────────
        self.extractors = nn.ModuleList([
            ScaleLevelExtractor(
                in_channels, feat_channels,
                dilation=self.dilations[k],
                stride=self.strides[k],
            )
            for k in range(num_levels)
        ])

        # ── Capsule routers: between adjacent levels (fine → coarse) ──────────
        # Router k routes from level k into level k+1
        self.routers = nn.ModuleList([
            SpatialCapsuleRouter(feat_channels, neighborhood, num_routing_iters)
            for _ in range(num_levels - 1)
        ])

        # ── Project every level to the coarsest spatial resolution ────────────
        # Level k has grid_sizes[k] = N/2^k.
        # Coarsest has grid_sizes[K-1] = N/2^(K-1).
        # Reduction ratio for level k = 2^(K-1-k).
        coarsest = self.grid_sizes[-1]
        self.projections = nn.ModuleList()
        for k in range(num_levels):
            ratio  = self.grid_sizes[k] // coarsest     # always a power of 2
            log2r  = int(math.log2(ratio)) if ratio > 1 else 0
            self.projections.append(_make_cascade_proj(feat_channels, log2r))

        # ── Fuse all projected levels ─────────────────────────────────────────
        g = min(8, self.Cout)
        self.fusion = nn.Sequential(
            nn.Conv3d(feat_channels * num_levels, self.Cout, 1, bias=False),
            nn.GroupNorm(g, self.Cout),
            nn.GELU(),
        )
        self.final_pool = nn.AvgPool3d(4, 4)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input volume, shape ``(B, in_channels, N, N, N)``.

        Returns:
            Fused multi-scale features,
            shape ``(B, out_channels, N//2^(K-1), N//2^(K-1), N//2^(K-1))``.
        """
        # ── 1. Multi-scale feature extraction ────────────────────────────────
        # feats[k]: (B, C, grid_sizes[k], grid_sizes[k], grid_sizes[k])
        feats: List[torch.Tensor] = [ext(x) for ext in self.extractors]

        # ── 2. Capsule routing: bottom-up propagation (fine → coarse) ─────────
        # Each router augments the coarser level with aggregated fine-level info.
        routed: List[torch.Tensor] = list(feats)
        for k, router in enumerate(self.routers):
            routed[k + 1] = router(routed[k], routed[k + 1])

        # ── 3. Project all levels to coarsest spatial resolution ──────────────
        projected = [proj(f) for proj, f in zip(self.projections, routed)]

        # ── 4. Concatenate across levels and fuse ─────────────────────────────
        return self.final_pool(self.fusion(torch.cat(projected, dim=1)))

    # ─────────────────────────────────────────────────────────────────────────

    def routing_topology(self) -> List[torch.Tensor]:
        """
        Return the learned coupling-prior probability vectors for every router.

        Each entry has shape ``(hood³,)`` and sums to 1.  At init all entries
        equal 1/hood³; after training the distribution reflects learned routing
        structure.

        Returns:
            List of length K-1; entry k corresponds to the router that sends
            features from level k to level k+1.

        Example::

            topo = layer.routing_topology()
            for k, probs in enumerate(topo):
                print(f"Router {k}→{k+1}: {probs.detach().cpu().numpy().round(3)}")
        """
        return [r.routing_weights() for r in self.routers]

    # ─────────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        header = (
            f"volume_side={self.N}, num_levels={self.K}, "
            f"feat_channels={self.C}, out_channels={self.Cout}\n"
            f"  {'Level':>5}  {'Stride':>6}  {'Dilation':>8}  "
            f"{'Grid pts/dim':>12}  {'Total pts':>12}"
        )
        rows = [header]
        for k in range(self.K):
            g = self.grid_sizes[k]
            rows.append(
                f"  {k:>5}  {self.strides[k]:>6}  {self.dilations[k]:>8}  "
                f"{g:>12}  {g**3:>12,}"
            )
        return "\n".join(rows)

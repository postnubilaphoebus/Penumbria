# -*- coding: utf-8 -*-
# File    : filter_response_normalization.py
# Author : tattaka
# Email  : tattaka666@gmail.com
# Date    : 27/11/2019 (Modified for FRN3d on 03/12/2025)
#
# This file is part of Filter-Response-Normalization-PyTorch
# https://github.com/tattaka/Filter-Response-Normalization-PyTorch
# Distributed under MIT License.

import torch
from torch import nn
import torch.nn.functional as F

# Assuming 'replicate' and 'DataParallelWithCallback' are defined elsewhere
# For a standalone file, you might need to adjust or remove this import
from .replicate import DataParallelWithCallback 

__all__ = [
    'FilterResponseNorm1d', 'FilterResponseNorm2d', 'FilterResponseNorm3d', 'convert_model' # ADDED 3d
]

## Big fix for https://github.com/tattaka/Filter-Response-Normalization-PyTorch/issues/2
class _FilterResponseNorm(nn.Module):
    # ADDED 'tau', 'beta', 'gamma' to __constants__ for clarity
    __constants__ = ["num_features", "eps", "eps_trainable", "tau", "beta", "gamma"] 

    def __init__(self, shape, activated=True, eps=1e-6, eps_trainable=True):
        super(_FilterResponseNorm, self).__init__()
        self._eps = eps
        self.activated = activated
        self.num_features = shape[1] # Channels dimension is always the second dim
        self.eps_trainable = eps_trainable

        # The shape for the learned parameters must match the expected output shape
        # after unsqueezing/broadcasting in the forward pass.
        self.beta = nn.Parameter(torch.zeros(shape))
        self.gamma = nn.Parameter(torch.ones(shape))

        if self.eps_trainable:
            self.eps = nn.Parameter(torch.full(shape, eps))
        else:
            self.eps = eps

        if self.activated:
            self.tau = nn.Parameter(torch.zeros(shape))
        else:
            self.tau = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        if isinstance(self.eps, nn.Parameter):
            nn.init.constant_(self.eps, self._eps)
        if self.tau is not None:
            nn.init.zeros_(self.tau)

    def _check_input_dim(self, input):
        raise NotImplementedError


class FilterResponseNorm1d(_FilterResponseNorm):

    def __init__(self, num_features, activated=True, eps=1e-6, eps_trainable=True):
        super(FilterResponseNorm1d, self).__init__(
            shape=(1, num_features), # (1, C)
            activated=activated,
            eps=eps,
            eps_trainable=eps_trainable,
        )
        
    def forward(self, input):
        self._check_input_dim(input)
        
        # Original implementation for 1D seems to calculate nu2 differently:
        # nu2 = input.pow(2) 
        # The line below (commented out in your source) is the actual FRN nu2 calculation, 
        # but for 1D the implementation is simplified:
        # nu2 = torch.mean(input.pow(2), axis=1, keepdims=True)
        # We will keep the original implementation's behavior for 1D.
        nu2 = input.pow(2) 
        
        # Original FRN: L2 norm over spatial dims and channels, but here it's just element-wise square
        # For simplicity and adherence to the original code's forward pass:
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps) + self._eps)
        output = self.gamma * input + self.beta
        
        if self.activated:
            output = torch.max(output, self.tau)
            
        return output
    
    def _check_input_dim(self, input):
        if input.dim() != 2: # Expected shape: (N, C) or (C, L)
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))


class FilterResponseNorm2d(_FilterResponseNorm):

    def __init__(self, num_features, activated=True, eps=1e-6, eps_trainable=True):
        super(FilterResponseNorm2d, self).__init__(
            shape=(1, num_features, 1, 1), # (1, C, 1, 1)
            activated=activated,
            eps=eps,
            eps_trainable=eps_trainable,
        )
        
    def forward(self, input):
        self._check_input_dim(input)
        # N, C, H, W -> Calculate mean over H and W (dims 2 and 3)
        nu2 = torch.mean(input.pow(2), dim=[2, 3], keepdim=True)
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps) + self._eps)
        output = self.gamma * input + self.beta
        if self.activated:
            output = torch.max(output, self.tau)
        return output
    
    def _check_input_dim(self, input):
        if input.dim() != 4: # Expected shape: (N, C, H, W)
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


# --- NEW 3D IMPLEMENTATION ---
class FilterResponseNorm3d(_FilterResponseNorm):

    def __init__(self, num_features, activated=True, eps=1e-6, eps_trainable=True):
        super(FilterResponseNorm3d, self).__init__(
            shape=(1, num_features, 1, 1, 1), # (1, C, 1, 1, 1)
            activated=activated,
            eps=eps,
            eps_trainable=eps_trainable,
        )
        
    def forward(self, input):
        self._check_input_dim(input)
        
        # N, C, D, H, W -> Calculate mean over D, H, and W (dims 2, 3, and 4)
        # FRN computes the L2 norm across all spatial and batch dimensions, 
        # but keeps the channel dimension separate.
        nu2 = torch.mean(input.pow(2), dim=[2, 3, 4], keepdim=True)
        
        # Normalization step: L_norm = a / sqrt(nu^2 + tau)
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps) + self._eps)
        
        # TLU (Thresholded Linear Unit) and affine transformation (Gamma * L_norm + Beta)
        output = self.gamma * input + self.beta
        
        if self.activated:
            # TLU is applied element-wise, max(x, tau)
            output = torch.max(output, self.tau)
            #output = torch.max(output, self.tau) + 0.1 * torch.min(output, self.tau)
            
        return output
    
    def _check_input_dim(self, input):
        if input.dim() != 5: # Expected shape: (N, C, D, H, W)
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
# --- END NEW 3D IMPLEMENTATION ---


class FilterResponseNorm3dMish(_FilterResponseNorm):
    def __init__(self, num_features, activated=True, eps=1e-6, eps_trainable=True):
        super(FilterResponseNorm3dMish, self).__init__(
            shape=(1, num_features, 1, 1, 1), 
            activated=activated,
            eps=eps,
            eps_trainable=eps_trainable,
        )

    def forward(self, input):
        self._check_input_dim(input)

        # 1. Compute the mean square (nu^2) over spatial dims (D, H, W)
        nu2 = torch.mean(input.pow(2), dim=[2, 3, 4], keepdim=True)

        # 2. Normalize: input / sqrt(nu^2 + eps)
        # We use torch.abs(self.eps) to ensure stability if eps_trainable is True
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps) + self._eps)

        # 3. Affine transformation: y = gamma * x + beta
        output = self.gamma * input + self.beta

        # 4. Thresholded Mish Activation
        if self.activated:
            # Shift the signal by the learnable threshold tau
            shifted_output = output - self.tau
            
            # Mish = x * tanh(softplus(x))
            # Applying Mish to the threshold-shifted values
            output = output * torch.tanh(F.softplus(shifted_output))

        return output

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f'expected 5D input (got {input.dim()}D input)')


class GatedContextFRN3d(nn.Module):
    def __init__(self, num_features, kernel_size=3, eps=1e-6, init_bias=-5.0):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # FRN Parameters
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.sigma = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))

        self.gate_conv = nn.Conv3d(num_features, num_features, kernel_size=1)
        
        with torch.no_grad():
            self.gate_conv.weight.fill_(0.01) 
            self.gate_conv.bias.fill_(init_bias) 

        # This will store the TV loss for the current forward pass
        self.last_tv_loss = 0.0

    def compute_3d_tv(self, gate):
        # gate shape: [B, C, D, H, W]
        C = gate.shape[1]
        kernel_data = torch.tensor([-1.0, 1.0], dtype=gate.dtype, device=gate.device)
        diff_kernel = kernel_data.view(1, 1, 1, 1, 2)
        
        # 2. Expand creates a view (no extra memory) instead of .repeat()
        expanded_kernel = diff_kernel.expand(C, 1, 1, 1, 2)
        
        # 3. Compute differences using groups=C (Depthwise)
        # Width differences
        tv_w = F.conv3d(gate, expanded_kernel, groups=C).abs().mean()
        
        # Height differences
        tv_h = F.conv3d(gate, expanded_kernel.transpose(-1, -2), groups=C).abs().mean()
        
        # Depth differences
        tv_d = F.conv3d(gate, expanded_kernel.transpose(-1, -3), groups=C).abs().mean()
        
        return tv_w + tv_h + tv_d

    def forward(self, x):
        self._check_input_dim(x)
        
        nu2_global = torch.mean(x.pow(2), dim=[2, 3, 4], keepdim=True)
        nu2_local = F.avg_pool3d(x.pow(2), kernel_size=self.kernel_size, stride=1, padding=self.padding)
        
        # 3. Compute the Spatial Gate
        relative_energy = nu2_local / (nu2_global + self.eps)
        gate = torch.sigmoid(self.gate_conv(relative_energy))
        
        # Store for the optimizer to pick up or for logging
        self.last_tv_loss = self.compute_3d_tv(gate)
        
        # 4. Spatially Adaptive Mixing
        mixed_nu2 = (gate * nu2_local) + ((1 - gate) * nu2_global)
        
        # 5. Normalize and TLU
        x_norm = x * torch.rsqrt(mixed_nu2 + torch.abs(self.sigma) + self.eps)
        output = self.gamma * x_norm + self.beta
        
        return torch.max(output, self.tau)

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f'expected 5D input (got {input.dim()}D input)')




def convert_model(module):
    """Traverse the input module and its child recursively
        and replace all instance of torch.nn.modules.batchnorm.BatchNorm*N*d + ReLU()
        to FilterResponseNorm*N*d
    """

    mod = module
    
    # 1D/2D conversion (existing logic)
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
        mod = FilterResponseNorm1d(module.num_features, activated=True, eps=module.eps)
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        mod = FilterResponseNorm2d(module.num_features, activated=True, eps=module.eps)
        
    # ADDED 3D conversion logic
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm3d):
        mod = FilterResponseNorm3d(module.num_features, activated=True, eps=module.eps)
        
    elif isinstance(module, torch.nn.ReLU):
        mod = torch.nn.Identity()
        
    # Recursive conversion for child modules
    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))
        
    return mod

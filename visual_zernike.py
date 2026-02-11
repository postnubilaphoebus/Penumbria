import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.data
import skimage.transform
from scipy.ndimage import distance_transform_edt

# ------------------------------------------------------------
# 1. Input: horse silhouette, rotated, EDT
# ------------------------------------------------------------
horse = skimage.data.horse()
horse = skimage.transform.resize(horse, (64, 64), anti_aliasing=False)
horse = horse < 0.5
horse = np.rot90(horse, 2)                     # 180° rotation – upright
edt = distance_transform_edt(horse)
input_img = edt / edt.max()
size = input_img.shape[0]

# ------------------------------------------------------------
# 2. Fourier transform and frequency coordinates
# ------------------------------------------------------------
F = np.fft.fft2(input_img)
F_shift = np.fft.fftshift(F)
mag = np.log(np.abs(F_shift) + 1)

fx = np.fft.fftshift(np.fft.fftfreq(size))
fy = np.fft.fftshift(np.fft.fftfreq(size))
FX, FY = np.meshgrid(fx, fy)
rho = np.sqrt(FX**2 + FY**2)
rho = np.clip(rho, 0, 1)

# ------------------------------------------------------------
# 3. Zernike modes (j=3, j=4, j=12)
# ------------------------------------------------------------
Z3 = FY / np.max(np.abs(FY))
Z4 = 2 * rho**2 - 1
Z12 = 6 * rho**4 - 6 * rho**2 + 1

# Extreme coefficients – unmistakable distortion for reviewers
alpha3 =  2.0    # tilt – strong translation
alpha4 =  2.0    # defocus – strong blurring / contrast inversion
alpha12 = -2.0   # spherical – strong edge ringing / halo

# ------------------------------------------------------------
# 4. Phase mask and application
# ------------------------------------------------------------
phase_mask = alpha3 * Z3 + alpha4 * Z4 + alpha12 * Z12
phase_mask_wrapped = np.angle(np.exp(1j * phase_mask))   # wrap to [-π,π]

phase_factor = np.exp(1j * phase_mask)
F_aberrated = F_shift * phase_factor
# magnitude unchanged – not plotted

# ------------------------------------------------------------
# 5. Output image (normalised for display)
# ------------------------------------------------------------
output_img = np.fft.ifft2(np.fft.ifftshift(F_aberrated)).real
output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())

# ------------------------------------------------------------
# 6. Tall, clean 2×3 layout – perfectly spaced and aligned
# ------------------------------------------------------------
fig = plt.figure(figsize=(16, 11), dpi=150)
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3,
              left=0.06, right=0.94, top=0.78, bottom=0.08)

# ---- (1) Input image ----
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(input_img, cmap='gray', origin='lower', extent=(-1,1,-1,1))
ax1.set_title('(1) Input image\nEDT of horse', fontweight='bold', fontsize=14)
ax1.set_xlabel('x', fontsize=12); ax1.set_ylabel('y', fontsize=12)
# Hidden colorbar – forces same width as (4)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad=0.04)
cax1.set_visible(False)

# ---- (2) Fourier magnitude ----
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(mag, cmap='inferno', origin='lower', extent=(-0.5,0.5,-0.5,0.5))
ax2.set_title('(2) Fourier magnitude\n(log, shifted)', fontweight='bold', fontsize=14)
ax2.set_xlabel('fx', fontsize=12); ax2.set_ylabel('fy', fontsize=12)
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes("right", size="5%", pad=0.04)
plt.colorbar(im2, cax=cax2, ticks=[0, 2, 4])

# ---- (3) Zernike modes (combined panel) ----
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('(3) Zernike modes', fontweight='bold', fontsize=14)
ax3.axis('off')

w, h = 0.24, 0.4
spacing = 0.03
x0, y0 = 0.05, 0.55

# j=3 – leftmost: full labels
iax1 = ax3.inset_axes([x0, y0, w, h])
im_j3 = iax1.imshow(Z3, cmap='RdBu', origin='lower', extent=(-1,1,-1,1), vmin=-1, vmax=1)
iax1.set_title('j=3', fontsize=11)
iax1.set_xlabel('fx', fontsize=9)
iax1.set_ylabel('fy', fontsize=9)
iax1.tick_params(labelsize=8)

# j=4 – no y‑axis labels
iax2 = ax3.inset_axes([x0 + w + spacing, y0, w, h])
im_j4 = iax2.imshow(Z4, cmap='RdBu', origin='lower', extent=(-1,1,-1,1), vmin=-1, vmax=1)
iax2.set_title('j=4', fontsize=11)
iax2.set_xlabel('fx', fontsize=9)
iax2.set_ylabel('')
iax2.tick_params(labelsize=8, left=False, labelleft=False)

# j=12 – no y‑axis labels
iax3 = ax3.inset_axes([x0 + 2*(w + spacing), y0, w, h])
im_j12 = iax3.imshow(Z12, cmap='RdBu', origin='lower', extent=(-1,1,-1,1), vmin=-1, vmax=1)
iax3.set_title('j=12', fontsize=11)
iax3.set_xlabel('fx', fontsize=9)
iax3.set_ylabel('')
iax3.tick_params(labelsize=8, left=False, labelleft=False)

# Colourbar – far right, cleanly separated
cax = ax3.inset_axes([x0 + 3*(w + spacing) + 0.02, y0, 0.02, h])
cbar = plt.colorbar(im_j12, cax=cax, ticks=[-1, 0, 1])
cbar.ax.tick_params(labelsize=8)
cbar.set_label('amplitude', fontsize=10)

# ---- (4) Weighted sum ----
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(phase_mask, cmap='RdBu', origin='lower', extent=(-1,1,-1,1))
ax4.set_title('(4) Weighted sum\nα₃·Z3 + α₄·Z4 + α₁₂·Z12', fontweight='bold', fontsize=12)
ax4.set_xlabel('fx', fontsize=12); ax4.set_ylabel('fy', fontsize=12)
div4 = make_axes_locatable(ax4)
cax4 = div4.append_axes("right", size="5%", pad=0.04)
cbar4 = plt.colorbar(im4, cax=cax4)
cbar4.ax.tick_params(labelsize=10)

# ---- (5) Phase mask (wrapped) ----
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.imshow(phase_mask_wrapped, cmap='twilight', origin='lower',
                 extent=(-1,1,-1,1), vmin=-np.pi, vmax=np.pi)
ax5.set_title('(5) Phase mask\nwrapped to [-π,π]', fontweight='bold', fontsize=14)
ax5.set_xlabel('fx', fontsize=12); ax5.set_ylabel('fy', fontsize=12)
div5 = make_axes_locatable(ax5)
cax5 = div5.append_axes("right", size="5%", pad=0.04)
cbar5 = plt.colorbar(im5, cax=cax5)
cbar5.ax.tick_params(labelsize=10)
cbar5.set_label('phase [rad]', fontsize=11)

# ---- (6) Output image ----
ax6 = fig.add_subplot(gs[1, 2])
im6 = ax6.imshow(output_img, cmap='gray', origin='lower', extent=(-1,1,-1,1))
ax6.set_title('(6) Output image\n(inverse FFT, normalised)', fontweight='bold', fontsize=14)
ax6.set_xlabel('x', fontsize=12); ax6.set_ylabel('y', fontsize=12)
div6 = make_axes_locatable(ax6)
cax6 = div6.append_axes("right", size="5%", pad=0.04)
cbar6 = plt.colorbar(im6, cax=cax6)
cbar6.ax.tick_params(labelsize=10)

# ---- Overall title – perfectly elevated ----
fig.suptitle('GlobalZernikeConv3d Layer – Conceptual 2D Analogue', 
             fontsize=18, fontweight='bold', y=0.94)

plt.tight_layout()
plt.show()
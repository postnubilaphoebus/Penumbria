import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skimage.data
import skimage.transform
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift

# ------------------------------------------------------------
# Create a cleaner input - simpler shape with sharper features
# ------------------------------------------------------------
# Create a simple cell-like blob instead of horse
size = 128
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

# Create an elliptical "cell" with some internal structure - SHARPER
mask = ((X/0.6)**2 + (Y/0.4)**2) < 1
nucleus = ((X/0.2)**2 + (Y/0.2)**2) < 1

input_img = np.zeros((size, size))
input_img[mask] = 0.6
input_img[nucleus] = 1.0

# Add only very slight smoothing to keep it sharp
input_img = gaussian_filter(input_img, sigma=0.8)

# ------------------------------------------------------------
# Simulate different aberration effects simply
# ------------------------------------------------------------

def simulate_longitudinal_shift(img):
    """Simulate axial defocus - blur"""
    return gaussian_filter(img, sigma=4)

def simulate_lateral_shift(img):
    """Simulate lateral shift"""
    return shift(img, shift=(5, 3), mode='constant', cval=0)

def simulate_defocus(img):
    """Simulate defocus - strong blur"""
    return gaussian_filter(img, sigma=6)

def simulate_spherical(img):
    """Simulate spherical aberration - edge halos"""
    from scipy.ndimage import sobel
    # Create edge-emphasized version
    edges = np.sqrt(sobel(img, axis=0)**2 + sobel(img, axis=1)**2)
    # Add halo around edges
    halo = gaussian_filter(edges, sigma=4)
    return np.clip(img + 0.8*halo - 0.3*edges, 0, 1)

# ------------------------------------------------------------
# Create simple visualization
# ------------------------------------------------------------
fig = plt.figure(figsize=(18, 12), dpi=150)
gs = GridSpec(2, 3, figure=fig, 
              hspace=0.4, wspace=0.3,
              left=0.06, right=0.94, top=0.86, bottom=0.06)

# Grid overlay function
def add_grid(ax, size, gridlines=8):
    """Add a grid overlay to show deformations"""
    step = size // gridlines
    for i in range(0, size, step):
        ax.axhline(i, color='cyan', alpha=0.4, linewidth=0.8)
        ax.axvline(i, color='cyan', alpha=0.4, linewidth=0.8)

# Simplified aberration examples
examples = [
    ("No Aberration\n(Clear)", input_img, "Original clear image"),
    ("Lateral Shift (j=1,2)\nTranslation", simulate_lateral_shift(input_img), "Image shifted in x-y"),
    ("Defocus (j=4)\nBlurring", simulate_defocus(input_img), "Out-of-focus blur"),
    ("Longitudinal Shift (j=3)\nAxial defocus", simulate_longitudinal_shift(input_img), "Axial displacement"),
    ("Spherical (j=12)\nEdge halos", simulate_spherical(input_img), "Edge artifacts"),
]

# Plot in grid - use all 6 positions for symmetry
positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]

for idx, (title, img, description) in enumerate(examples):
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=1)
    
    # Add grid overlay
    add_grid(ax, size)
    
    ax.set_title(title, fontweight='bold', fontsize=18, pad=10)
    ax.set_xlabel(description, fontsize=15, style='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add scale bar
    scalebar_length = size // 5
    ax.plot([10, 10 + scalebar_length], [10, 10], 'w-', linewidth=3)
    ax.text(10 + scalebar_length//2, 20, '20 μm', color='white', 
            fontsize=10, ha='center', fontweight='bold')

# Add one more example in the 6th position
ax6 = fig.add_subplot(gs[1, 2])
# Show combined aberration
combined = simulate_defocus(simulate_lateral_shift(input_img))
ax6.imshow(combined, cmap='gray', origin='lower', vmin=0, vmax=1)
add_grid(ax6, size)
ax6.set_title("Combined\nShift + Defocus", fontweight='bold', fontsize=18, pad=10)
ax6.set_xlabel("Multiple aberrations", fontsize=15, style='italic')
ax6.set_xticks([])
ax6.set_yticks([])
scalebar_length = size // 5
ax6.plot([10, 10 + scalebar_length], [10, 10], 'w-', linewidth=3)
ax6.text(10 + scalebar_length//2, 20, '20 μm', color='white', 
        fontsize=10, ha='center', fontweight='bold')

# Overall title
fig.suptitle('GlobalZernikeConv3d Layer: Optical Aberration Correction on Synthetic Cell', 
             fontsize=22, fontweight='bold', y=0.96)

plt.savefig('/mnt/user-data/outputs/zernike_simple_intuitive.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("Simple visualization saved!")

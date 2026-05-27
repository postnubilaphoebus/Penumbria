import numpy as np
import taichi as ti
from scipy.ndimage import label
import skimage
from skimage.segmentation import watershed

ti.init(arch=ti.gpu)

@ti.kernel
def compute_directions(
    vol: ti.types.ndarray(dtype=ti.f32, ndim=3),
    direction: ti.types.ndarray(dtype=ti.u8, ndim=3)
):
    offsets = ti.Matrix([
        (-1,-1,-1),(-1,-1, 0),(-1,-1, 1),
        (-1, 0,-1),(-1, 0, 0),(-1, 0, 1),
        (-1, 1,-1),(-1, 1, 0),(-1, 1, 1),
        ( 0,-1,-1),( 0,-1, 0),( 0,-1, 1),
        ( 0, 0,-1),            ( 0, 0, 1),
        ( 0, 1,-1),( 0, 1, 0),( 0, 1, 1),
        ( 1,-1,-1),( 1,-1, 0),( 1,-1, 1),
        ( 1, 0,-1),( 1, 0, 0),( 1, 0, 1),
        ( 1, 1,-1),( 1, 1, 0),( 1, 1, 1),
    ], dt=ti.f32)
        
    distances = ti.Matrix([
        ti.sqrt(3.), ti.sqrt(2.), ti.sqrt(3.),
        ti.sqrt(2.), 1.,          ti.sqrt(2.),
        ti.sqrt(3.), ti.sqrt(2.), ti.sqrt(3.),
        ti.sqrt(2.), 1.,          ti.sqrt(2.),
        1.,                       1.,
        ti.sqrt(2.), 1.,          ti.sqrt(2.),
        ti.sqrt(3.), ti.sqrt(2.), ti.sqrt(3.),
        ti.sqrt(2.), 1.,          ti.sqrt(2.),
        ti.sqrt(3.), ti.sqrt(2.), ti.sqrt(3.),
    ], dt=ti.f32)

    for z, y, x in ti.ndrange(*vol.shape):
        center = vol[z, y, x]
        min_slope = 0.0
        best_dir = ti.u8(255)
        
        for i in ti.static(range(26)):
            dz = ti.cast(offsets[i, 0], ti.i32)
            dy = ti.cast(offsets[i, 1], ti.i32)
            dx = ti.cast(offsets[i, 2], ti.i32)
            
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < vol.shape[0] and \
               0 <= ny < vol.shape[1] and \
               0 <= nx < vol.shape[2]:
                slope = (center - vol[nz, ny, nx]) / distances[i]
                if slope > min_slope:
                    min_slope = slope
                    best_dir = ti.u8(i)
        direction[z, y, x] = best_dir

@ti.kernel
def compute_directions_larger(
    vol: ti.types.ndarray(dtype=ti.f32, ndim=3),
    direction: ti.types.ndarray(dtype=ti.u8, ndim=3)
):
    _offsets = [(dx, dy, dz)
            for dx in (-3, 0, 3)
            for dy in (-3, 0, 3)
            for dz in (-3, 0, 3)
            if (dx, dy, dz) != (0, 0, 0)]

    offsets   = ti.Matrix(_offsets, dt=ti.f32)
    distances = ti.Matrix([(dx*dx + dy*dy + dz*dz)**0.5 for dx, dy, dz in _offsets],
                        dt=ti.f32)

    for z, y, x in ti.ndrange(*vol.shape):
        center = vol[z, y, x]
        min_slope = 0.0
        best_dir = ti.u8(255)
        
        for i in ti.static(range(26)):
            dz = ti.cast(offsets[i, 0], ti.i32)
            dy = ti.cast(offsets[i, 1], ti.i32)
            dx = ti.cast(offsets[i, 2], ti.i32)
            
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < vol.shape[0] and \
               0 <= ny < vol.shape[1] and \
               0 <= nx < vol.shape[2]:
                slope = (center - vol[nz, ny, nx]) / distances[i]
                if slope > min_slope:
                    min_slope = slope
                    best_dir = ti.u8(i)
        direction[z, y, x] = best_dir

@ti.kernel
def rainfall_full_path(
    direction: ti.types.ndarray(dtype=ti.u8, ndim=3),
    seed_ids:  ti.types.ndarray(dtype=ti.i32, ndim=3),
    labels:    ti.types.ndarray(dtype=ti.i32, ndim=3),
):
    offsets = ti.Matrix([
        (-1,-1,-1),(-1,-1, 0),(-1,-1, 1),
        (-1, 0,-1),(-1, 0, 0),(-1, 0, 1),
        (-1, 1,-1),(-1, 1, 0),(-1, 1, 1),
        ( 0,-1,-1),( 0,-1, 0),( 0,-1, 1),
        ( 0, 0,-1),            ( 0, 0, 1),
        ( 0, 1,-1),( 0, 1, 0),( 0, 1, 1),
        ( 1,-1,-1),( 1,-1, 0),( 1,-1, 1),
        ( 1, 0,-1),( 1, 0, 0),( 1, 0, 1),
        ( 1, 1,-1),( 1, 1, 0),( 1, 1, 1),
    ], dt=ti.f32)
    
    dz = ti.Matrix([o[0] for o in offsets], dt=ti.i32)
    dy = ti.Matrix([o[1] for o in offsets], dt=ti.i32)
    dx = ti.Matrix([o[2] for o in offsets], dt=ti.i32)
    for z, y, x in ti.ndrange(*labels.shape):
        if seed_ids[z, y, x] != 0:
            labels[z, y, x] = seed_ids[z, y, x]
            continue
        cz, cy, cx = z, y, x
        for _ in range(600):
            d = ti.cast(direction[cz, cy, cx], ti.i32)
            if d == 255:
                break
            nz = cz + dz[d]
            ny = cy + dy[d]
            nx = cx + dx[d]
            if nz < 0 or nz >= labels.shape[0] or \
               ny < 0 or ny >= labels.shape[1] or \
               nx < 0 or nx >= labels.shape[2]:
                break
            if seed_ids[nz, ny, nx] != 0:
                labels[z, y, x] = seed_ids[nz, ny, nx]
                break
            cz, cy, cx = nz, ny, nx

def run_rainfall(direction, seed_ids):
    labels = seed_ids.copy()
    rainfall_full_path(direction, seed_ids, labels)
    return labels

def fast_wts(img, seed_ids, bg_thres):
    direction_mask = (img < bg_thres).astype(bool)
    invert_img = 1.0 - img
    directions = np.zeros_like(invert_img, dtype=np.uint8)
    compute_directions(invert_img, directions)
    directions[direction_mask] = 255
    labels = run_rainfall(directions, seed_ids)
    return labels


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from tqdm import tqdm
    img = skimage.io.imread("8_inference_output.tif")
    seed_ids, _ = label(img > 0.6)
    #import pdb; pdb.set_trace()
    def bench_once(img):
        seed_ids, _ = label(img > 0.6)

        # warmup for the GPU path (first call usually includes JIT/alloc)
        _ = fast_wts(img, seed_ids, 0.1)

        t0 = time.perf_counter()
        _ = fast_wts(img, seed_ids, 0.1)
        t1 = time.perf_counter()
        _ = watershed(-img, markers=seed_ids, mask=img > 0.1)
        t2 = time.perf_counter()

        return t1 - t0, t2 - t1

    scales = np.linspace(0.1, 2.0, 12)
    voxels, gpu_times, cpu_times = [], [], []
    num_bench = 0
    for s in tqdm(scales):
        new_shape = tuple(max(1, int(round(d * s))) for d in img.shape)
        img_s = resize(img, new_shape, order=1, preserve_range=True,
                    anti_aliasing=False).astype(img.dtype)
        t_gpu, t_cpu = bench_once(img_s)
        n = np.prod(new_shape)
        voxels.append(n)
        gpu_times.append(t_gpu)
        cpu_times.append(t_cpu)
        print(f"scale={s:.2f}  shape={new_shape}  voxels={n/1e6:.1f}M  "
            f"gpu={t_gpu:.3f}s  cpu={t_cpu:.3f}s")
        num_bench += 1

    voxels = np.array(voxels) / 1e6  # millions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(voxels, gpu_times, 'o-', label='fast_wts (GPU)')
    ax.plot(voxels, cpu_times, 's-', label='skimage watershed (CPU)')
    ax.set_xlabel('Voxels (millions)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Watershed benchmark vs image size')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

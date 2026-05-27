from skimage.segmentation import watershed
from scipy.ndimage import label, find_objects, gaussian_filter
import numpy as np
from matching import matching
import numba as nb
import SimpleITK as sitk
from scipy.ndimage import sobel
from watershed_fast import fast_wts

def gradient_symmetry_voting(image, r=5):
    gx = sobel(image, axis=0)
    gy = sobel(image, axis=1)
    gz = sobel(image, axis=2)
    return vote_map_3d(gx, gy, gz, r)


@nb.njit(parallel=True)
def vote_map_3d(gx, gy, gz, r):
    shape = gx.shape
    out = np.zeros(shape, dtype=np.float32)
    eps = 1e-8

    for x in nb.prange(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # gradient vector
                gxv = gx[x, y, z]
                gyv = gy[x, y, z]
                gzv = gz[x, y, z]

                mag = np.sqrt(gxv**2 + gyv**2 + gzv**2)
                if mag > eps:
                    # unit gradient
                    ux = gxv / mag
                    uy = gyv / mag
                    uz = gzv / mag

                    # positive vote
                    cx = int(round(x + ux * r))
                    cy = int(round(y + uy * r))
                    cz = int(round(z + uz * r))
                    if 0 <= cx < shape[0] and 0 <= cy < shape[1] and 0 <= cz < shape[2]:
                        out[cx, cy, cz] += 1

                    # negative vote
                    cx = int(round(x - ux * r))
                    cy = int(round(y - uy * r))
                    cz = int(round(z - uz * r))
                    if 0 <= cx < shape[0] and 0 <= cy < shape[1] and 0 <= cz < shape[2]:
                        out[cx, cy, cz] -= 1

    return out

def muti_scale_symmetry(img, background_threshold = 0.05):
    background_mask = img > background_threshold
    res_map = []
    for i in range(7, 14, 2):
        res = gradient_symmetry_voting(img, i)
        res = gaussian_filter(res, sigma=0.5)
        res = (res - res.min()) / (res.max() - res.min())
        res_map.append(res)
    res_map = np.array(res_map)
    res_map = np.mean(res_map, axis=0)
    res_map[~background_mask] = 0
    return res_map
    
def has_nonzero_elements(lst):
    if all(isinstance(el, list) for el in lst):
        return any(any(sublist) for sublist in lst)
    else:
        return any(lst)
    
def twice_smooth_and_threshold(img):
    gfilt = gaussian_filter(img, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    gfilt = gaussian_filter(gfilt, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    return gfilt

def watershed_inference(prediction, 
                        padding,
                        minimum_cell_size = 9,
                        h = 0.1, 
                        cell_confidence_minimum = 0.5, 
                        background_threshold = 0.2,
                        gaussian_smoothing = True,
                        simple_thresholding = False,
                        low_confidence_merging = False,
                        sym = False):
    
    if padding is not None and has_nonzero_elements(padding):
        shape_trimming_by_extra_width = True
    else:
        shape_trimming_by_extra_width = False
    prediction = prediction.astype(np.float32)
    hdome_image = prediction.copy()

    if gaussian_smoothing:
        hdome_image = gaussian_filter(hdome_image, sigma=1)
        if hdome_image.max() != hdome_image.min():
            hdome_image = (hdome_image - hdome_image.min()) / (hdome_image.max() - hdome_image.min() + 1e-8)
        else:
            hdome_image = np.zeros_like(hdome_image)
    if hdome_image.sum() == 0:
        if shape_trimming_by_extra_width:
            if prediction.ndim == 3:
                return np.zeros_like(prediction)[padding[0][0]:prediction.shape[0]-padding[0][1], 
                                                 padding[1][0]:prediction.shape[1]-padding[1][1], 
                                                 padding[2][0]:prediction.shape[2]-padding[2][1]]
            elif prediction.ndim == 2:
                return np.zeros_like(prediction)[padding[0][0]:prediction.shape[0]-padding[0][1], 
                                                 padding[1][0]:prediction.shape[1]-padding[1][1]]
            else:
                raise ValueError(f"incorrect image shape{prediction.shape}")
        else:
            return np.zeros_like(prediction)
        
    if not simple_thresholding:
        # hdome transform
        if sym:
            hdome_image = muti_scale_symmetry(hdome_image)
        image_sitk = sitk.GetImageFromArray(hdome_image)
        marker_image = sitk.Subtract(image_sitk, h)
        reconstructed = sitk.ReconstructionByDilation(marker_image, image_sitk)
        reconstructed = sitk.GetArrayFromImage(reconstructed)
        h_maxima = hdome_image - reconstructed
        h_maxima = h_maxima * (prediction > h)
        h_maxima_binary = h_maxima > 0
        labeled_array, _ = label(h_maxima_binary)
    else:
        h_maxima_binary = hdome_image > h
        labeled_array, _ = label(h_maxima_binary)


    # watershed flooding
    background_image = (prediction > background_threshold).astype(int)
    wts = watershed(-prediction, labeled_array, mask = background_image)
    ## you can change to GPU watershed if you like, works until 600M voxels with 8GB VRAM
    # wts = fast_wts(prediction, labeled_array, background_threshold) 

    # cell confidence exclusion
    slices = find_objects(wts)
    wts_numpy_array = np.zeros_like(wts)
    idx = 1
    for i, slice_tuple in enumerate(slices, start=1):
        if slice_tuple is not None:
            local_locs = np.array(np.where(wts[slice_tuple] == i))
            global_locs = np.stack(local_locs).T + np.array([s.start for s in slice_tuple])
            if len(global_locs) > minimum_cell_size:
                if prediction.ndim == 3:
                    values = prediction[global_locs[:, 0], global_locs[:, 1], global_locs[:, 2]]
                elif prediction.ndim == 2:
                    values = prediction[global_locs[:, 0], global_locs[:, 1]]
                else:
                    raise ValueError(f"incorrect image shape{prediction.shape}")
                ind = np.argpartition(values, -1)[-1:]
                top_1 = values[ind]
                if np.all(top_1 > cell_confidence_minimum):
                    if prediction.ndim == 3:
                        wts_numpy_array[global_locs[:, 0], global_locs[:, 1], global_locs[:, 2]] = idx
                    elif prediction.ndim == 2:
                        wts_numpy_array[global_locs[:, 0], global_locs[:, 1]] = idx
                    else:
                        raise ValueError(f"incorrect image shape{prediction.shape}")
                    idx += 1

    return wts_numpy_array

def objective(trial, img, target, padding_list_val, data_dimensionality):
    gaussian_smoothing = trial.suggest_categorical('gaussian_smoothing', [True, False])
    h = trial.suggest_float('h', 0.15, 0.5, step=0.01)
    bg = trial.suggest_float('bg', 0.05, 0.25, step=0.01)
    c = trial.suggest_float('c', 0.2, 0.75, step=0.01)
    simple_thresholding = trial.suggest_categorical('simple_thresholding', [True, False])
    sym = False
    lcm = False
    sum_acc = 0
    map_vals = np.arange(0.1, 1.0, 0.1) if data_dimensionality == 3 else np.arange(0.5, 1.0, 0.05)
    for im, ta, pa in zip(img, target, padding_list_val):
        try:
            pred = watershed_inference(
                prediction=im,
                padding = pa,
                minimum_cell_size=9,
                h=h,
                cell_confidence_minimum=c,
                background_threshold=bg,
                gaussian_smoothing=gaussian_smoothing,
                simple_thresholding=simple_thresholding,
                low_confidence_merging=lcm,
                sym=sym
            )
            mean_acc = 0
            idx = 0
            for val in map_vals:
                stats_dict = matching(ta, pred, val)
                mean_acc += stats_dict.accuracy
                idx += 1
            sum_acc += (mean_acc / idx)
        except:
            sum_acc += 0
        
    return sum_acc / len(img)

if __name__ == "__main__":
    pass


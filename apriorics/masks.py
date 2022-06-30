from functools import partial
from multiprocessing import Array, Pool
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from pathaia.util.types import NDBoolMask, NDByteGrayImage, NDImage
from PIL.Image import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hed, rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    disk,
    label,
    remove_small_holes,
    remove_small_objects,
)


def remove_large_objects(ar, max_size=1000, connectivity=1):
    out = ar.copy()

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_large = component_sizes > max_size
    too_large_mask = too_large[ccs]
    out[too_large_mask] = 0
    return out


def get_tissue_mask(img_G: NDByteGrayImage, blacktol: int = 0, whitetol: int = 247):
    r"""
    Get basic tissue mask from grayscale image.

    Args:
        img_G: grayscale input image as numpy byte array.
        blacktol: minimum value to be considered foreground.
        whitetol: maximul value to be considered foreground.

    Returns:
        Mask as a boolean array.
    """
    return (img_G > blacktol) & (img_G < whitetol)


def get_dab_mask(
    ihc: NDImage,
    dab_thr: float = 0.03,
    object_min_size: int = 1000,
    hole_min_size: int = 1000,
    tissue_mask: Optional[NDBoolMask] = None,
    he: Optional[NDImage] = None,
    binary_op: Optional[Callable] = None,
    r: int = 10,
) -> NDBoolMask:
    r"""
    Computes a mask using otsu thresholding on DAB channel on an immunohistochemistry
    image.

    Args:
        ihc: input immunohistochemistry image.
        dab_thr: minimum value to use for thresholding.
        object_min_size: the smallest allowable object size.
        hole_min_size: the smallest allowable hole size.
        tissue_mask: mask to use on input image.
        he: input H&E image. If specified, pixels where DAB value is high for both `ihc`
            and `he` will be negative. This can be used for artifact detection.
        binary_op: must be a binary operation from `skimage.morphology` (dilation,
            erosion, opening or closing).
        r: radius of the disk to use as footprint for `binary_op`

    Returns:
        DAB thresholded mask.
    """
    img_dab = rgb2hed(ihc)[:, :, 2]

    if tissue_mask is not None:
        img_dab = np.ma.masked_array(img_dab, mask=tissue_mask)
    thr = max(
        threshold_otsu(image=img_dab, hist=None),
        dab_thr,
    )

    mask = img_dab > thr
    if tissue_mask is not None:
        mask = mask & tissue_mask
    if he is not None:
        # check for artifacts
        mask = mask & (rgb2hed(he)[:, :, 2] < thr)

    mask = remove_small_objects(mask, min_size=object_min_size)

    if binary_op is not None:
        mask |= binary_op(mask, footprint=disk(r))

    mask = remove_small_holes(mask, area_threshold=hole_min_size)

    return mask


def get_mask_ink(img):
    mask = np.ones(img.shape[:2], dtype=bool)
    ranges = [(56, 98), (69, 119), (117, 160)]
    for c, r in enumerate(ranges):
        a, b = r
        mask &= (img[..., c] > a) & (img[..., c] < b)
    mask = binary_dilation(remove_small_objects(mask, min_size=50), disk(15))
    return mask


def get_mask_AE1AE3(
    he: Union[Image, NDImage], ihc: Union[Image, NDImage]
) -> NDBoolMask:
    r"""
    Compute mask on paired AE1AE3 immunohistochemistry and H&E images.

    Args:
        he: input H&E image. Mask is computed using a threshold on H channel.
        ihc: input immunohistochemistry image. Mask is computed using a threshold on
            DAB channel.

    Returns:
        Intersection of H&E and IHC masks.
    """
    he = np.asarray(he)
    he_H = rgb2hed(he)
    he_DAB = he_H[:, :, 2]
    he_H = he_H[:, :, 0]
    he_hue = rgb2hsv(he)[:, :, 0]
    ihc = np.asarray(ihc)
    ihc_DAB = rgb2hed(ihc)[:, :, 2]
    ihc_s = rgb2hsv(ihc)[:, :, 1]
    mask_he = binary_closing(
        remove_small_objects(
            remove_small_holes((he_H > 0.005) & (he_hue > 0.69), area_threshold=1000),
            min_size=500,
        ),
        footprint=disk(10),
    )
    mask_ihc = binary_closing(
        remove_small_objects(
            remove_small_holes((ihc_DAB > 0.03) & (ihc_s > 0.1), area_threshold=1000),
            min_size=500,
        ),
        footprint=disk(10),
    )
    mask_he_DAB = binary_closing(
        remove_small_objects(
            remove_small_holes(he_DAB > 0.03, area_threshold=1000), min_size=500
        ),
        footprint=disk(10),
    )
    mask = remove_small_objects(mask_he & ~mask_he_DAB & mask_ihc, min_size=500)
    return mask


def get_mask_PHH3(he: Union[Image, NDImage], ihc: Union[Image, NDImage]) -> NDBoolMask:
    r"""
    Compute mask on paired PHH3 immunohistochemistry and H&E images.

    Args:
        he: input H&E image. Mask is computed using a threshold on H channel.
        ihc: input immunohistochemistry image. Mask is computed using a threshold on
            DAB channel.

    Returns:
        Intersection of H&E and IHC masks.
    """
    he = np.asarray(he)
    he_H = rgb2hed(he)
    he_DAB = he_H[:, :, 2]
    he_H = he_H[:, :, 0]
    he_hue = rgb2hsv(he)[:, :, 0]
    ihc = np.asarray(ihc)
    ihc_DAB = rgb2hed(ihc)[:, :, 2]
    ihc_s = rgb2hsv(ihc)[:, :, 1]

    mask_he1 = (he_H > 0.06) & (he_H < 0.14) & (he_hue > 0.67)
    mask_he2 = get_mask_ink(he)
    mask_he = remove_small_objects(
        remove_small_holes(mask_he1 & ~mask_he2, area_threshold=50),
        min_size=50,
    )

    mask_ihc = remove_small_objects(
        remove_small_holes((ihc_DAB > 0.04) & (ihc_s > 0.1), area_threshold=50),
        min_size=50,
    )

    mask_he_DAB = remove_small_objects(
        remove_small_holes(he_DAB > 0.04, area_threshold=50), min_size=50
    )

    mask = remove_large_objects(
        remove_small_objects(mask_he & mask_ihc & ~mask_he_DAB, min_size=50),
        max_size=1000,
    )
    return mask


def get_mask_function(ihc_type: str) -> Callable:
    r"""
    Get mask function corresponding to an immunohistochemistry type.

    Args:
        ihc_type: name of the IHC technique.

    Returns:
        Corresponding masking function.
    """
    return globals()[f"get_mask_{ihc_type}"]


def update_full_mask(
    full_mask: NDBoolMask,
    mask: NDBoolMask,
    x: int,
    y: int,
):
    r"""
    Update a portion of a large mask using a smaller mask.

    Args:
        full_mask: large mask to update.
        mask: small mask to use for update.
        x: x coordinate of top-left corner of `mask` on `full_mask`.
        y: y coordinate of top-left corner of `mask` on `full_mask`.
    """
    h, w = full_mask.shape
    p_h, p_w = mask.shape
    dy = min(h, y + p_h) - y
    dx = min(w, x + p_w) - x
    full_mask[y : y + dy, x : x + dx] = mask[:dy, :dx]


def update_full_mask_mp(
    full_mask: Array, mask: NDBoolMask, x: int, y: int, w: int, h: int
):
    r"""
    Update a portion of a large mask using a smaller mask.

    Args:
        full_mask: large mask to update.
        mask: small mask to use for update.
        x: x coordinate of top-left corner of `mask` on `full_mask`.
        y: y coordinate of top-left corner of `mask` on `full_mask`.
    """
    p_h, p_w = mask.shape
    dy = min(h, y + p_h) - y
    dx = min(w, x + p_w) - x
    for i in range(y, y + dy):
        full_mask[i * w + x : i * w + x + dx] = mask[i - y, :dx]


def merge_bboxes(
    bboxes: List[List[int]], masks: List[NDBoolMask]
) -> Tuple[List[List[int]], List[NDBoolMask]]:
    n = len(bboxes)
    for i in range(1, n):
        for j in range(i):
            if bboxes[j] is None:
                continue
            xi0, yi0, xi1, yi1 = bboxes[i]
            xj0, yj0, xj1, yj1 = bboxes[j]

            x0, y0, x1, y1 = min(xi0, xj0), min(yi0, yj0), max(xi1, xj1), max(yi1, yj1)
            h = y1 - y0
            w = x1 - x0
            maski = np.zeros((h, w), dtype=bool)
            maski[yi0 - y0 : yi1 - y0, xi0 - x0 : xi1 - x0] = 1
            maskj = np.zeros((h, w), dtype=bool)
            maskj[yj0 - y0 : yj1 - y0, xj0 - x0 : xj1 - x0] = 1

            if (maski & maskj).sum():
                bboxes[i] = [min(xi0, xj0), min(yi0, yj0), max(xi1, xj1), max(yi1, yj1)]
                masks[i] = masks[i] | masks[j]

                bboxes[j] = None
                masks[j] = None
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    masks = [mask for mask in masks if mask is not None]
    return bboxes, masks


def mask_to_bbox(mask: Union[NDBoolMask, torch.Tensor], pad: int = 5, min_size=10):
    tensor = isinstance(mask, torch.Tensor)
    labels, n = label(mask, return_num=True)
    bboxes = []
    masks = []
    h, w = mask.shape

    for i in range(1, n + 1):
        mask = labels == i
        if mask.sum() < min_size:
            continue
        ii, jj = np.nonzero(mask)
        y0, y1 = ii.min(), ii.max()
        x0, x1 = jj.min(), jj.max()
        bboxes.append(
            [
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(w - 1, x1 + pad),
                min(h - 1, y1 + pad),
            ]
        )
        masks.append(mask)
    bboxes, masks = merge_bboxes(bboxes, masks)
    bboxes, masks = np.array(bboxes, dtype=np.float32), np.stack(masks).astype(np.uint8)
    if tensor:
        bboxes, masks = torch.as_tensor(bboxes), torch.as_tensor(masks)
    return bboxes, masks


def decode_rle(rle, x, y, w, h):
    mask = np.zeros((h, w), dtype=bool)
    for i in range(y, y + h):
        rle_row = rle[i]
        for k in range(0, len(rle_row), 2):
            j0 = rle_row[k] - x
            j1 = j0 + rle_row[k + 1]
            mask[i - y, max(0, j0) : min(j1, w)] = True
    return mask


def _encode_rle_row(w, i):
    mask_row = mask[i * w : (i + w) * w]
    prev = False
    rle_row = []
    for k, v in enumerate(mask_row):
        if v:
            if prev:
                rle_row[-1] += 1
            else:
                rle_row.extend([k % w, 1])
                prev = True
        else:
            prev = False
    return i, rle_row


def encode_rle(mask, w, h, num_workers=None):
    rle = [[]] * h

    def _init_func(mask_arg):
        global mask
        mask = mask_arg

    with Pool(processes=num_workers, initializer=_init_func, initargs=(mask,)) as pool:
        rle_rows = pool.map(partial(_encode_rle_row, w), range(h))
        pool.close()
        pool.join()
    for i, rle_row in rle_rows:
        rle[i] = rle_row
    return rle


def update_rle(rle, mask, x, y, slide_w):
    h, w = mask.shape

    old_mask = decode_rle(rle, x, y, w, h)
    mask |= old_mask

    for i in range(h):
        prev = False
        rle_row = []
        for k, v in enumerate(mask[i]):
            if v:
                if prev:
                    rle_row[-1] += 1
                else:
                    rle_row.extend([(k + x) % slide_w, 1])
                    prev = True
            else:
                prev = False
        rle[y + i] = rle_row

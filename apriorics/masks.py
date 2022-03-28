from multiprocessing import Array
from typing import Callable, Optional, Union, List, Tuple
from skimage.filters import threshold_otsu
from skimage.color import rgb2hed, rgb2hsv
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_closing,
    disk,
    label,
)
import numpy as np
from pathaia.util.types import NDImage, NDByteGrayImage, NDBoolMask
from PIL.Image import Image
from scipy import ndimage as ndi
import torch


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
    mask_he = remove_small_objects(
        remove_small_holes(
            (he_H > 0.06) & (he_H < 0.14) & (he_hue > 0.69), area_threshold=50
        ),
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

            h = max(yi1, yj1) - min(yi0, yj0)
            w = max(xi1, xj1) - min(xi0, xj0)
            maski = np.zeros((h, w), dtype=bool)
            maski[yi0:yi1, xi0:xi1] = 1
            maskj = np.zeros((h, w), dtype=bool)
            maskj[yj0:yj1, xj0:xj1] = 1

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

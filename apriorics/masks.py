from typing import Callable, Optional, Union
from skimage.filters import threshold_otsu
from skimage.color import rgb2hed
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_closing,
    disk,
)
import numpy as np
from pathaia.util.types import NDImage, NDByteGrayImage, NDBoolMask
from PIL.Image import Image


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
    he_H = rgb2hed(np.asarray(he))[:, :, 0]
    ihc_DAB = rgb2hed(np.asarray(ihc))[:, :, 2]
    mask_he = binary_closing(
        remove_small_objects(
            remove_small_holes(he_H > 0.005, area_threshold=1000), min_size=500
        ),
        footprint=disk(10),
    )
    mask_ihc = binary_closing(
        remove_small_objects(
            remove_small_holes(ihc_DAB > 0.03, area_threshold=1000), min_size=500
        ),
        footprint=disk(10),
    )
    mask = remove_small_objects(mask_he & mask_ihc, min_size=500)
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

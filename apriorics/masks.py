from skimage.filters import threshold_otsu
from skimage.color import rgb2hed
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_closing,
    disk,
)
import numpy as np


def get_tissue_mask(img_G, blacktol=0, whitetol=247):
    """
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
    ihc,
    dab_thr=0.085,
    object_min_size=1000,
    hole_min_size=1000,
    tissue_mask=None,
    he=None,
    binary_op=None,
    r=10,
):
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
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        selem = x ** 2 + y ** 2 <= r ** 2
        mask |= binary_op(mask, selem=selem)

    mask = remove_small_holes(mask, area_threshold=hole_min_size)

    return mask


def get_mask_AE1AE3(he, ihc):
    he_H = rgb2hed(np.array(he))[:, :, 0]
    ihc_DAB = rgb2hed(np.array(ihc))[:, :, 2]
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


def get_mask_function(ihc_type):
    return globals()[f"get_mask_{ihc_type}"]

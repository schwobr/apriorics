from skimage.color import rgb2gray, rgb2hed
from skimage.util import img_as_ubyte, img_as_float
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from subprocess import run
from pathlib import Path


def get_input_images(slide, patch, h_min, h_max):
    img = np.asarray(
        slide.read_region(location=patch.position, level=patch.level, size=patch.size)
    )
    img_G = img_as_ubyte(rgb2gray(img))
    img_H = img_as_ubyte((rgb2hed(img) - h_min) / (h_max - h_min))
    return img, img_G, img_H


def get_tissue_mask(img_G, blacktol=0, whitetol=247):
    return (img_G > blacktol) & (img_G < whitetol)


def has_enough_tissue(he_G, ihc_G, blacktol=0, whitetol=247, area_thr=0.99):
    size = he_G.size
    area_thr = area_thr * size
    mask1 = get_tissue_mask(he_G)
    mask2 = get_tissue_mask(ihc_G)
    return (mask1.sum() > area_thr) and (mask2.sum() > area_thr)


def get_dab_mask(
    img,
    dab_thr=0.085,
    object_min_size=1000,
    tissue_mask=None,
    he_img=None,
    binary_op=None,
    r=10,
):
    img_dab = rgb2hed(img)[:, :, 2]

    if tissue_mask is not None:
        img_dab = np.ma.masked_array(img_dab, mask=tissue_mask)
    thr = max(
        threshold_otsu(image=img_dab, hist=None),
        dab_thr,
    )

    mask = img_dab > thr
    if tissue_mask is not None:
        mask = mask & tissue_mask
    if he_img is not None:
        # check for artifacts
        mask = mask & (he_img < thr)

    mask = remove_small_objects(mask, min_size=object_min_size)

    if binary_op is not None:
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        selem = x ** 2 + y ** 2 <= r ** 2
        mask |= binary_op(mask, selem=selem)

    mask = remove_small_holes(mask, area_threshold=object_min_size)

    return mask


def equalize_contrasts(
    he_H, ihc_H, he_G, ihc_G, whitetol=220, low_percentile=5, high_percentile=95
):
    he_H = img_as_float(he_H)
    ihc_H = img_as_float(ihc_H)
    ihc_min = np.percentile(ihc_H, low_percentile)
    ihc_max = np.percentile(ihc_H, high_percentile)
    for img, img_g in zip((he_H, ihc_H), (he_G, ihc_G)):
        img[:] = ((img - ihc_min) / (ihc_max - ihc_min)).clip(0, 1)
        img[img_g > whitetol] = 0
    return img_as_ubyte(he_H), img_as_ubyte(ihc_H)


def convert_to_nifti(path):
    path = Path(path)
    base_dir = path.parent
    path = Path(f"/data/inputs/{path.name}")
    c2d_cmd = (
        f"docker run -v {base_dir}:/data/inputs historeg "
        f"c2d -mcs {path} -foreach -orient LP -spacing 1x1mm -origin 0x0mm -endfor -omc"
        f" {path.with_suffix('.nii.gz')}"
    )
    run(c2d_cmd.split())


def register(
    base_path,
    he_H_path,
    ihc_H_path,
    he_path,
    ihc_path,
    reg_path,
    iterations=20000,
):
    he_H_path, ihc_H_path, he_path, ihc_path, reg_path = map(
        lambda x: Path("/data/inputs") / x.relative_to(base_path),
        (he_H_path, ihc_H_path, he_path, ihc_path, reg_path),
    )
    historeg_cmd = (
        f"docker run -v {base_path}:/data/inputs historeg "
        f"HistoReg -i {iterations} -f {he_H_path} -m {ihc_H_path} -o "
        f"{base_path/'historeg'}"
    )
    run(historeg_cmd.split())

    tfm_path = (
        base_path
        / f"historeg/{he_H_path.stem}_registered_to_{ihc_H_path.stem}"
        / "metrics/full_resolution"
    )
    greedy_cmd = (
        f"greedy -d 2 -rf {he_path} -rm {ihc_path} {reg_path} -r "
        f"{tfm_path/'big_warp.nii.gz'} {tfm_path/'Affine.mat'}"
    )
    run(greedy_cmd.split())

    c2d_cmd = (
        f"c2d -mcs {reg_path} -foreach -type uchar -endfor -omc "
        f"{reg_path.with_suffix('.png')}"
    )
    run(c2d_cmd.split())

from skimage.color import rgb2hed
import cv2
from skimage.util import img_as_ubyte, img_as_float
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_closing,
    binary_dilation,
)
from skimage.measure import label
from subprocess import run
from pathlib import Path


def get_input_images(slide, patch, h_min, h_max):
    img = np.asarray(
        slide.read_region(location=patch.position, level=patch.level, size=patch.size)
    )
    img_G = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_H = img_as_ubyte((rgb2hed(img) - h_min) / (h_max - h_min))
    return img, img_G, img_H


def get_tissue_mask(img_G, blacktol=0, whitetol=247):
    return (img_G > blacktol) & (img_G < whitetol)


def has_enough_tissue(he_G, ihc_G, blacktol=0, whitetol=247, area_thr=0.99):
    size = he_G.size
    area_thr = area_thr * size
    mask1 = get_tissue_mask(he_G, blacktol=blacktol, whitetol=whitetol)
    mask2 = get_tissue_mask(ihc_G, blacktol=blacktol, whitetol=whitetol)
    return (mask1.sum() > area_thr) and (mask2.sum() > area_thr)


def get_binary_op(op_name):
    if op_name == "closing":
        return binary_closing
    elif op_name == "dilation":
        return binary_dilation
    else:
        return None


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
        f"docker run -v {base_path}:/data/inputs historeg "
        f"greedy -d 2 -rf {he_path} -rm {ihc_path} {reg_path} -r "
        f"{tfm_path/'big_warp.nii.gz'} {tfm_path/'Affine.mat'}"
    )
    run(greedy_cmd.split())

    c2d_cmd = (
        f"docker run -v {base_path}:/data/inputs historeg "
        f"c2d -mcs {reg_path} -foreach -type uchar -endfor -omc "
        f"{reg_path.with_suffix('.png')}"
    )
    run(c2d_cmd.split())


def get_dot_mask(slide, thumb_level=3, min_val=60, max_val=90):
    thumb = np.array(
        slide.read_region(
            location=(0, 0),
            level=thumb_level,
            size=slide.resolutions["level_dimensions"][thumb_level],
        )
    )
    mask = np.ones(thumb.shape[:2], dtype=bool)
    for i in range(3):
        mask = mask & (min_val < thumb[:, :, i]) & (thumb[:, :, i] < max_val)
        mask = remove_small_holes(remove_small_objects(mask), area_threshold=500)
    return mask


def get_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    angle = np.arccos(dot) * np.sign((v2 - v1)[1])
    return angle


def get_sort_key(vertices):
    center = np.array(vertices).mean(0)

    def _key(vertex):
        vertex = np.array(vertex)
        vector = vertex - center
        return get_angle(np.array([1, 0]), vector)

    return _key


def get_vertices(mask):
    # penser au cas à plus ou moins de 4 points
    labels = label(mask)
    vertices = []
    for i in range(1, labels.max() + 1):
        ii, jj = (labels == i).nonzero()
        vertices.append((int(jj.mean()), int(ii.mean())))
    vertices.sort()
    while len(vertices) > 4:
        vertices.pop()
    vertices.sort(key=get_sort_key(vertices))
    return vertices


def get_rotation(fixed_vert, moving_vert):
    angle = get_angle(moving_vert[1] - moving_vert[0], fixed_vert[1] - fixed_vert[0])
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def get_scale(fixed_vert, moving_vert):
    x_moving_min, y_moving_min = moving_vert.min(0)
    x_moving_max, y_moving_max = moving_vert.max(0)
    x_fixed_min, y_fixed_min = fixed_vert.min(0)
    x_fixed_max, y_fixed_max = fixed_vert.max(0)
    if x_moving_max - x_moving_min < 5 or x_fixed_max - x_fixed_min < 5:
        scale_y = (y_fixed_max - y_fixed_min) / (y_moving_max - y_moving_min)
        scale_x = scale_y
    elif y_moving_max - y_moving_min < 5 or y_fixed_max - y_fixed_min < 5:
        scale_x = (x_fixed_max - x_fixed_min) / (x_moving_max - x_moving_min)
        scale_y = scale_x
    else:
        scale_x = (x_fixed_max - x_fixed_min) / (x_moving_max - x_moving_min)
        scale_y = (y_fixed_max - y_fixed_min) / (y_moving_max - y_moving_min)
    return np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])


def get_translation(fixed_vert, moving_vert):
    xoff, yoff = fixed_vert.min(0) - moving_vert.min(0)
    return np.array([[1, 0, xoff], [0, 1, yoff], [0, 0, 1]])


def equalize_vert_lengths(l1, l2):
    l1, l2 = sorted([l1, l2], key=len)
    sorted_l2 = sorted(
        l2, key=lambda x: min([(x[1] - y[1]) ** 2 + (x[0] - y[0]) ** 2 for y in l1])
    )
    while len(l2) > len(l1):
        to_remove = sorted_l2.pop()
        l2.remove(to_remove)


def get_affine_transform(fixed, moving):
    fixed_vert = get_vertices(fixed)
    moving_vert = get_vertices(moving)
    if len(fixed_vert) != len(moving_vert):
        equalize_vert_lengths(fixed_vert, moving_vert)
    fixed_vert = np.array(fixed_vert)
    moving_vert = np.array(moving_vert)
    rot = get_rotation(fixed_vert, moving_vert)
    reg_vert = np.concatenate(
        (moving_vert, [[1] for _ in range(len(fixed_vert))]), axis=1
    )
    reg_vert = (rot @ reg_vert.T).T
    scale = get_scale(fixed_vert, reg_vert[:, :2])
    reg_vert = (scale @ reg_vert.T).T
    trans = get_translation(fixed_vert, reg_vert[:, :2])
    return rot, scale, trans


def get_coord_transform(slide_he, slide_ihc):
    thumb_level = slide_he.resolutions["level_count"]
    mask_he = get_dot_mask(slide_he, thumb_level=thumb_level)
    mask_ihc = get_dot_mask(slide_ihc, thumb_level=thumb_level)
    rot, scale, trans = get_affine_transform(mask_ihc, mask_he)
    dsr = slide_he.shape[0] / mask_he.shape[0]
    trans[:2, 2] *= dsr
    affine = trans @ scale @ rot

    def _transform(x, y):
        x1, y1, _ = (affine @ np.array([x, y, 1]).T).T
        return x1, y1

    return _transform

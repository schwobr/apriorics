import csv
from os import PathLike
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from albumentations import BasicTransform, Compose
from pathaia.util.basic import ifnone
from pathaia.util.types import Patch, Slide
from scipy.sparse import load_npz
from torch.utils.data import Dataset, RandomSampler, Sampler
from tqdm import tqdm

from apriorics.masks import mask_to_bbox
from apriorics.transforms import StainAugmentor


class SegmentationDataset(Dataset):
    r"""
    PyTorch dataset for slide segmentation tasks.

    Args:
        slide_paths: list of slides' filepaths.
        mask_paths: list of masks' filepaths. Masks are supposed to be tiled pyramidal
            images.
        patches_paths: list of patch csvs' filepaths. Files must be formatted according
            to `PathAIA API <https://github.com/MicroMedIAn/PathAIA>`_.
        stain_matrices_paths: path to stain matrices .npy files. Each file must contain
            a (2, 3) matrice to use for stain separation. If not sppecified while
            `stain_augmentor` is, stain matrices will be computed at runtime (can cause
            a bottleneckd uring training).
        stain_augmentor: :class:`~apriorics.transforms.StainAugmentor` object to use for
            stain augmentation.
        transforms: list of `albumentation <https://albumentations.ai/>`_ transforms to
            use on images (and on masks when relevant).
        slide_backend: whether to use `OpenSlide <https://openslide.org/>`_ or
            `cuCIM <https://github.com/rapidsai/cucim>`_ to load slides.
    """

    def __init__(
        self,
        slide_paths: Sequence[PathLike],
        mask_paths: Sequence[PathLike],
        patches_paths: Sequence[PathLike],
        stain_matrices_paths: Optional[Sequence[PathLike]] = None,
        stain_augmentor: Optional[StainAugmentor] = None,
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
    ):
        super().__init__()
        self.slides = []
        self.masks = []
        self.patches = []
        self.slide_idxs = []
        self.n_pos = []

        for slide_idx, (patches_path, slide_path, mask_path) in enumerate(
            zip(patches_paths, slide_paths, mask_paths)
        ):
            self.slides.append(Slide(slide_path, backend=slide_backend))
            self.masks.append(Slide(mask_path, backend=slide_backend))
            with open(patches_path, "r") as patch_file:
                reader = csv.DictReader(patch_file)
                for patch in reader:
                    self.patches.append(Patch.from_csv_row(patch))
                    self.n_pos.append(patch["n_pos"])
                    self.slide_idxs.append(slide_idx)

        self.n_pos = np.array(self.n_pos, dtype=np.uint64)

        if stain_matrices_paths is None:
            self.stain_matrices = None
        else:
            self.stain_matrices = [np.load(path) for path in stain_matrices_paths]
        self.stain_augmentor = stain_augmentor
        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]
        slide_idx = self.slide_idxs[idx]
        slide = self.slides[slide_idx]
        mask = self.masks[slide_idx]

        slide_region = np.asarray(
            slide.read_region(patch.position, patch.level, patch.size).convert("RGB")
        )
        mask_region = np.asarray(
            mask.read_region(patch.position, patch.level, patch.size).convert("1"),
            dtype=np.float32,
        )

        if self.stain_augmentor is not None:
            if self.stain_matrices is not None:
                stain_matrix = self.stain_matrices[slide_idx]
            else:
                stain_matrix = None
            slide_region = self.stain_augmentor(image=(slide_region, stain_matrix))[
                "image"
            ]

        transformed = self.transforms(image=slide_region, mask=mask_region)
        return transformed["image"], transformed["mask"]


class SparseSegmentationDataset(Dataset):
    r"""
    PyTorch dataset for slide segmentation tasks.

    Args:
        slide_paths: list of slides' filepaths.
        mask_paths: list of masks' filepaths. Masks are supposed to be tiled pyramidal
            images.
        patches_paths: list of patch csvs' filepaths. Files must be formatted according
            to `PathAIA API <https://github.com/MicroMedIAn/PathAIA>`_.
        stain_matrices_paths: path to stain matrices .npy files. Each file must contain
            a (2, 3) matrice to use for stain separation. If not sppecified while
            `stain_augmentor` is, stain matrices will be computed at runtime (can cause
            a bottleneckd uring training).
        stain_augmentor: :class:`~apriorics.transforms.StainAugmentor` object to use for
            stain augmentation.
        transforms: list of `albumentation <https://albumentations.ai/>`_ transforms to
            use on images (and on masks when relevant).
        slide_backend: whether to use `OpenSlide <https://openslide.org/>`_ or
            `cuCIM <https://github.com/rapidsai/cucim>`_ to load slides.
    """

    def __init__(
        self,
        slide_paths: Sequence[PathLike],
        mask_paths: Sequence[PathLike],
        patches_paths: Sequence[PathLike],
        stain_matrices_paths: Optional[Sequence[PathLike]] = None,
        stain_augmentor: Optional[StainAugmentor] = None,
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
    ):
        super().__init__()
        self.slides = []
        self.masks = []
        self.patches = []
        self.slide_idxs = []
        self.n_pos = []

        for slide_idx, (patches_path, slide_path, mask_path) in enumerate(
            zip(patches_paths, slide_paths, mask_paths)
        ):
            self.slides.append(Slide(slide_path, backend=slide_backend))
            self.masks.append(load_npz(mask_path))
            with open(patches_path, "r") as patch_file:
                reader = csv.DictReader(patch_file)
                for patch in reader:
                    self.patches.append(Patch.from_csv_row(patch))
                    self.n_pos.append(patch["n_pos"])
                    self.slide_idxs.append(slide_idx)

        self.n_pos = np.array(self.n_pos, dtype=np.uint64)

        if stain_matrices_paths is None:
            self.stain_matrices = None
        else:
            self.stain_matrices = [np.load(path) for path in stain_matrices_paths]
        self.stain_augmentor = stain_augmentor
        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]
        slide_idx = self.slide_idxs[idx]
        slide = self.slides[slide_idx]
        mask = self.masks[slide_idx]

        slide_region = np.asarray(
            slide.read_region(patch.position, patch.level, patch.size).convert("RGB")
        )
        x, y = patch.position
        w, h = patch.size
        mask_region = mask[y : y + h, x : x + w].toarray().astype(np.float32)

        if self.stain_augmentor is not None:
            if self.stain_matrices is not None:
                stain_matrix = self.stain_matrices[slide_idx]
            else:
                stain_matrix = None
            slide_region = self.stain_augmentor(image=(slide_region, stain_matrix))[
                "image"
            ]

        transformed = self.transforms(image=slide_region, mask=mask_region)
        return transformed["image"], transformed["mask"]


class DetectionDataset(Dataset):
    r"""
    PyTorch dataset for slide segmentation tasks.

    Args:
        slide_paths: list of slides' filepaths.
        mask_paths: list of masks' filepaths. Masks are supposed to be tiled pyramidal
            images.
        patches_paths: list of patch csvs' filepaths. Files must be formatted according
            to `PathAIA API <https://github.com/MicroMedIAn/PathAIA>`_.
        stain_matrices_paths: path to stain matrices .npy files. Each file must contain
            a (2, 3) matrice to use for stain separation. If not sppecified while
            `stain_augmentor` is, stain matrices will be computed at runtime (can cause
            a bottleneckd uring training).
        stain_augmentor: :class:`~apriorics.transforms.StainAugmentor` object to use for
            stain augmentation.
        transforms: list of `albumentation <https://albumentations.ai/>`_ transforms to
            use on images (and on masks when relevant).
        slide_backend: whether to use `OpenSlide <https://openslide.org/>`_ or
            `cuCIM <https://github.com/rapidsai/cucim>`_ to load slides.
    """

    def __init__(
        self,
        slide_paths: Sequence[PathLike],
        mask_paths: Sequence[PathLike],
        patches_paths: Sequence[PathLike],
        stain_matrices_paths: Optional[Sequence[PathLike]] = None,
        stain_augmentor: Optional[StainAugmentor] = None,
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
        min_size: int = 10,
    ):
        super().__init__()
        self.slides = []
        self.masks = []
        self.patches = []
        self.slide_idxs = []
        self.n_pos = []

        for slide_idx, (patches_path, slide_path, mask_path) in enumerate(
            zip(patches_paths, slide_paths, mask_paths)
        ):
            self.slides.append(Slide(slide_path, backend=slide_backend))
            self.masks.append(Slide(mask_path, backend=slide_backend))
            with open(patches_path, "r") as patch_file:
                reader = csv.DictReader(patch_file)
                for patch in reader:
                    self.patches.append(Patch.from_csv_row(patch))
                    self.n_pos.append(patch["n_pos"])
                    self.slide_idxs.append(slide_idx)

        self.n_pos = np.array(self.n_pos, dtype=np.uint64)

        if stain_matrices_paths is None:
            self.stain_matrices = None
        else:
            self.stain_matrices = [np.load(path) for path in stain_matrices_paths]
        self.stain_augmentor = stain_augmentor
        self.transforms = Compose(ifnone(transforms, []))
        self.min_size = min_size
        self.clean()

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]
        slide_idx = self.slide_idxs[idx]
        slide = self.slides[slide_idx]
        mask = self.masks[slide_idx]

        slide_region = np.asarray(
            slide.read_region(patch.position, patch.level, patch.size).convert("RGB")
        )
        mask_region = np.asarray(
            mask.read_region(patch.position, patch.level, patch.size).convert("1"),
            dtype=np.uint8,
        )

        if self.stain_augmentor is not None:
            if self.stain_matrices is not None:
                stain_matrix = self.stain_matrices[slide_idx]
            else:
                stain_matrix = None
            slide_region = self.stain_augmentor(image=(slide_region, stain_matrix))[
                "image"
            ]

        retransform = True
        count = 0
        while retransform and count < 10:
            transformed = self.transforms(image=slide_region, mask=mask_region)
            retransform = transformed["mask"].sum() < self.min_size
            count += 1
        if retransform:
            return

        bboxes, masks = mask_to_bbox(transformed["mask"], pad=1, min_size=0)
        target = {
            "boxes": bboxes,
            "masks": masks,
            "labels": torch.ones(bboxes.shape[0], dtype=torch.int64),
        }
        return transformed["image"], target

    def clean(self):
        patches = []
        slide_idxs = []
        idxs = []
        for i in range(len(self)):
            if self[i] is not None:
                patches.append(self.patches[i])
                slide_idxs.append(self.slide_idxs[i])
                idxs.append(i)

        self.patches = patches
        self.slide_idxs = slide_idxs
        self.n_pos = self.n_pos[idxs]


class BalancedRandomSampler(RandomSampler):
    def __init__(
        self,
        data_source: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = False,
        generator: Optional[torch.Generator] = None,
        p_pos: float = 0.5,
    ):
        self.p_pos = p_pos
        super().__init__(
            data_source,
            replacement=replacement,
            num_samples=num_samples,
            generator=generator,
        )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        num_samples = super().num_samples
        if self.replacement:
            return num_samples
        else:
            n_pos = (self.data_source.n_pos > 0).sum()
            if n_pos == len(self.data_source):
                return num_samples
            else:
                max_avail = min(
                    int(n_pos / self.p_pos),
                    int((len(self.data_source) - n_pos) / (1 - self.p_pos)),
                )
            return min(num_samples, max_avail)

    def get_idxs(self) -> List[int]:
        mask = self.data_source.n_pos == 0
        avail = [mask.nonzero()[0].tolist(), (~mask).nonzero()[0].tolist()]
        avail = [cl_patches for cl_patches in avail if cl_patches]
        idxs = []
        num_samples = self.num_samples
        p = torch.rand(num_samples, 2, generator=self.generator)
        print("\nLoading sampler idxs...")
        for k in tqdm(range(num_samples), total=num_samples):
            if len(avail) == 1:
                cl_patches = avail[0]
                idx = torch.multinomial(torch.ones(len(cl_patches)), num_samples - k)
                idxs.extend([cl_patches[i] for i in idx])
                break
            x = p[k]
            cl = min(int(x[0] < self.p_pos), len(avail) - 1)
            cl_patches = avail[cl]
            idx = int(x[1] * len(cl_patches))
            if self.replacement:
                patch = cl_patches[idx]
            else:
                patch = cl_patches.pop(idx)
                if len(cl_patches) == 0:
                    avail.pop(cl)
            idxs.append(patch)
        assert len(idxs) == len(self)
        return np.random.permutation(idxs)

    def __iter__(self) -> Iterator[int]:
        return iter(self.get_idxs())

    def __len__(self) -> int:
        return self.num_samples


class ValidationPositiveSampler(Sampler):
    def __init__(
        self,
        data_source: Dataset,
        num_samples: Optional[int] = None,
    ):
        super().__init__(data_source)
        if num_samples is not None:
            assert num_samples <= len(data_source)
        self._num_samples = num_samples
        self.data_source = data_source

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is not None:
            return self._num_samples
        else:
            return (self.data_source.n_pos > 0).sum()

    def get_idxs(self) -> List[int]:
        n = self.num_samples
        mask = self.data_source.n_pos > 0
        pos_idxs = np.argwhere(mask).squeeze(1)
        idxs = pos_idxs[:n].tolist()
        if len(idxs) < n:
            neg_idxs = np.argwhere(~mask).squeeze(1)
            idxs.extend(neg_idxs[: n - len(idxs)].tolist())
        return idxs

    def __iter__(self) -> Iterator[int]:
        return iter(self.get_idxs())

    def __len__(self) -> int:
        return self.num_samples


class TestDataset(Dataset):
    def __init__(
        self,
        slide_path: PathLike,
        patches_path: PathLike,
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
    ):
        self.slide = Slide(slide_path, backend=slide_backend)
        self.patches = []
        with open(patches_path, "r") as patch_file:
            reader = csv.DictReader(patch_file)
            for patch in reader:
                self.patches.append(Patch.from_csv_row(patch))
        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        slide_region = np.asarray(
            self.slide.read_region(patch.position, patch.level, patch.size).convert(
                "RGB"
            )
        )

        transformed = self.transforms(image=slide_region)
        return transformed["image"]

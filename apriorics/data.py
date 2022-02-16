from os import PathLike
from typing import Sequence, Optional
import numpy as np
from torch.utils.data import Dataset
from pathaia.util.types import Slide, Patch
from pathaia.util.basic import ifnone
from albumentations import Compose, BasicTransform
import csv


class SegmentationDataset(Dataset):
    def __init__(
        self,
        slide_paths: Sequence[PathLike],
        mask_paths: Sequence[PathLike],
        patches_paths: Sequence[PathLike],
        stain_matrices_paths: Optional[Sequence[PathLike]] = None,
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
    ):
        super().__init__()
        self.slides = []
        self.masks = []
        self.patches = []
        self.slide_idxs = []

        for slide_idx, (patches_path, slide_path, mask_path) in enumerate(
            zip(patches_paths, slide_paths, mask_paths)
        ):
            self.slides.append(Slide(slide_path, backend=slide_backend))
            self.masks.append(Slide(mask_path, backend=slide_backend))
            with open(patches_path) as patch_file:
                reader = csv.DictReader(patch_file)
                for patch in reader:
                    self.patches.append(Patch.from_csv_row(patch))
                    self.slide_idxs.append(slide_idx)

        if stain_matrices_paths is None:
            self.stain_matrices = None
        else:
            self.stain_matrices = [np.load(path) for path in stain_matrices_paths]
        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        slide = self.slides[self.slide_idxs[idx]]
        mask = self.masks[self.slide_idxs[idx]]

        slide_region = np.asarray(
            slide.read_region(patch.location, patch.level, patch.size).convert("RGB")
        )
        mask_region = np.asarray(
            mask.read_region(patch.location, patch.level, patch.size).convert("1")
        )

        if self.stain_matrices is not None:
            slide_region.stain_matrix = self.stain_matrices[self.slide_idxs[idx]]

        transformed = self.transforms(
            image=np.asarray(slide_region), mask=np.asarray(mask_region)
        )
        return transformed["image"], transformed["mask"]

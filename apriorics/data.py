from os import PathLike
from typing import Sequence
import numpy as np
from torch.utils.data import Dataset
from pathaia.util.types import Slide, Patch
from pathaia.util.basic import ifnone
from albumentations import Compose
import csv


class SegmentationDataset(Dataset):
    def __init__(
        self,
        slide_paths: Sequence[PathLike],
        mask_paths: Sequence[PathLike],
        patches_paths: Sequence[PathLike],
        transforms=None,
        slide_backend="cucim",
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

        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        slide = self.slides[self.slide_idxs[idx]]
        mask = self.masks[self.slide_idxs[idx]]
        slide_region = slide.read_region(
            patch.location, patch.level, patch.size
        ).convert("RGB")
        mask_region = mask.read_region(patch.location, patch.level, patch.size).convert(
            "RGB"
        )
        transformed = self.transforms(
            image=np.asarray(slide_region), mask=np.asarray(mask_region)
        )
        return transformed["image"], transformed["mask"]

from os import PathLike
from typing import List, Tuple
from cv2 import transform
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from pathaia.util.types import Slide
from pathaia.util.basic import ifnone
from albumentations import Compose
import csv


SLIDES_BACKEND = "cucim"

class SegmentationDataset(Dataset):
    def __init__(self, slide_paths:List[PathLike], mask_paths:List[PathLike], patches_paths:List[PathLike], transforms=None):
        super().__init__()
        self.slides = [Slide(slide_path, backend=SLIDES_BACKEND) for slide_path in slide_paths]
        self.masks = [Slide(mask_path, backend=SLIDES_BACKEND) for mask_path in mask_paths]
        self.patches = []
        self.patches_slide = []
        for [slide_idx,patches_path] in enumerate(patches_paths):
            with open(patches_path) as patch_file : 
                reader = csv.DictReader(patch_file)
                for patch in reader :
                    self.patches.append(patch)
                    self.patches_slide.append(slide_idx)
        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        slide = self.slides[self.patches_slide[idx]]
        mask = self.masks[self.patches_slide[idx]]
        slide_region = slide.read_region([int(patch["x"]),int(patch["y"])], int(patch["level"]), [int(patch["size_x"]),int(patch["size_y"])])
        mask_region = mask.read_region([int(patch["x"]),int(patch["y"])], int(patch["level"]), [int(patch["size_x"]),int(patch["size_y"])])
        transformed = self.transforms(image=np.array(slide_region), mask=np.array(mask_region))
        return transformed["image"], transformed["mask"]

    
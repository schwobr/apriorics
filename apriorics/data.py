from os import PathLike
from typing import List, Tuple
from torch.utils.data import Dataset
from pathaia.util.types import Slide 
from pathaia.util.basic import ifnone
from albumentations import Compose
import csv


SLIDES_BACKEND = "cucim"
DEFAULT_PATCH_SIZE = (1024,1024)

class SegmentationDataset(Dataset):
    def __init__(self, slide_paths:List[PathLike], mask_paths:List[PathLike], patches_paths:List[PathLike], patch_size:Tuple[int,int] = DEFAULT_PATCH_SIZE, transforms=None):
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
        self.patch_size = patch_size
        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        slide = self.slides[self.patches_slide[idx]]
        mask = self.masks[self.patches_slide[idx]]
        return slide.read_region((patch["x"],patch["y"]), 0, self.patch_size), mask.read_region((patch["x"],patch["y"]), 0, self.patch_size)

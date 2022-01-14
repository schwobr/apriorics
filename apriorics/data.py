from torch.utils.data import Dataset
from cucim import CuImage
from pathaia.util.basic import ifnone
from albumentations import Compose


class SegmentationDataset(Dataset):
    def __init__(self, slide_paths, mask_paths, patches, transforms=None):
        super().__init__()
        self.slides = [CuImage(str(slide_path)) for slide_path in slide_paths]
        self.mask_rles = []
        for mask_path in mask_paths:
            with open(mask_path, "r") as f:
                self.mask_rles.append(f.read().rstrip())
        self.patches = patches
        self.transforms = Compose(ifnone(transforms, []))

    def __len__(self):
        return len(self.patch_coords)

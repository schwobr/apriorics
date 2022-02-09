from albumentations.core.transforms_interface import DualTransform
import numpy as np
from torchvision.transforms.functional import to_tensor


class ToSingleChanelMask (DualTransform):
    """
    Transforms that takes a grayscale masks with rgb or rgba chanels and transform them into a single channel image
    
    Target : mask, masks
    Type : any
    """

    def __init__(self, trailing_chanels:bool = True):
        super().__init__(True, 1)
        self.trailing_chanels = trailing_chanels

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img
    
    def apply_to_mask(self, img: np.ndarray, **params):
        if self.trailing_chanels: 
            return img[:,:,0]
        else:
            return img[0]

class DropAlphaChanel (DualTransform):
    """
    Transform that takes rgba images and mask and that removes the alpha chanel 
    
    Target : image, mask, masks
    Type : any
    """

    def __init__(self, trailing_chanels:bool = True):
        super().__init__(True, 1)
        self.trailing_chanels = trailing_chanels
    
    def apply(self, img: np.ndarray, **params):
        if self.trailing_chanels: 
            assert(img.shape[2] == 4)
            return img[:,:,:-1]
        else:
            assert(img.shape[0] == 4)
            return img[:-1]

class ToTensor (DualTransform):
    """
    Transform that takes np.ndarray and return a pytorch tensor
    
    Target : image, mask, masks
    Type : any
    """
    def __init__(self):
        super().__init__(True, 1)
    
    def apply(self, img: np.ndarray, **params):
        return to_tensor(img)

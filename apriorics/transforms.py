from albumentations.core.transforms_interface import DualTransform
import numpy as np


class ToSingleChannelMask(DualTransform):
    """
    Transforms that takes a grayscale masks with rgb or rgba channels and transform them
    into a single channel image

    Target : mask, masks
    Type : any
    """

    def __init__(self, trailing_channels: bool = True):
        super().__init__(True, 1)
        self.trailing_channels = trailing_channels

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img

    def apply_to_mask(self, img: np.ndarray, **params):
        if self.trailing_channels:
            return img[:, :, 0]
        else:
            return img[0]


class DropAlphaChannel(DualTransform):
    """
    Transform that takes rgba images and mask and that removes the alpha channel

    Target : image, mask, masks
    Type : any
    """

    def __init__(self, trailing_channels: bool = True):
        super().__init__(True, 1)
        self.trailing_channels = trailing_channels

    def apply(self, img: np.ndarray, **params):
        if self.trailing_channels:
            assert img.shape[2] == 4
            return img[:, :, :-1]
        else:
            assert img.shape[0] == 4
            return img[:-1]

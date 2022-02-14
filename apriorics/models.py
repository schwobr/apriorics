import numpy as np
from torch import nn
import torch
from torch.nn.functional import interpolate 
import timm
from apriorics.model_components.convolution import ConvBnRelu, LastCross, SelfAttentionBlock
from apriorics.model_components.decoder_blocks import DecoderBlock, PixelShuffleICNR
from apriorics.model_components.hooks import Hooks
from apriorics.model_components.normalization import bc_norm, group_norm
from apriorics.model_components.utils import get_sizes

class CBR(nn.Module):
    """"""

    def __init__(
        self,
        kernel_size,
        n_kernels,
        n_layers,
        n_classes=2,
        in_chans=3,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        in_c = in_chans
        out_c = n_kernels
        for k in range(n_layers):
            self.add_module(
                f"cbr{k}",
                ConvBnRelu(
                    in_c,
                    out_c,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    norm_layer=norm_layer,
                ),
            )
            # self.add_module(f'maxpool{k}', nn.MaxPool2d(3, stride=2, padding=1))
            in_c = out_c
            out_c *= 2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(out_c, n_classes)

    def forward(self, x):
        print(x.shape)
        for m in self.children():
            x = m(x)
        return x

class SASA(nn.Module):
    def __init__(self, kernel_size, n_kernels, n_layers, n_groups, n_classes=2, in_chans=3):
        super().__init__()

        self.stem = ConvBnRelu(
            in_chans, n_kernels, 7, stride=2, padding=3, padding_mode="reflect"
        )
        in_c = n_kernels
        out_c = 2 * n_kernels
        for k in range(n_layers):
            self.add_module(
                f"sasa_block_{k}",
                SelfAttentionBlock(
                    in_c, out_c, kernel_size, groups=n_groups
                ),
            )
            self.add_module(f"pool_{k}", nn.AvgPool2d(2, stride=2))
            in_c = out_c
            out_c *= 2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten(-2, -1)
        self.fc = nn.Linear(out_c, n_classes)

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x

class DynamicUnet(nn.Module):
    """"""

    def __init__(
        self, encoder_name, n_classes=2, input_shape=(3, 224, 224), pretrained=True
    ):
        super().__init__()
        norm_layer = group_norm
        if "cbr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=nn.BatchNorm2d)
            norm_layer = nn.BatchNorm2d
            cut = -3
        elif "cgr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=group_norm)
            cut = -3
        elif "bcr" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = CBR(*args, norm_layer=bc_norm)
            norm_layer = bc_norm
            cut = -3
        elif "sasa" in encoder_name:
            args = map(int, encoder_name.split("_")[1:])
            encoder = SASA(*args)
            cut = -3
        elif "sanet" in encoder_name:
            splits = encoder_name.split("_")
            kernel_size = int(splits[-1])
            encoder = globals()[splits[0]](kernel_size)
            cut = -2
        else:
            if encoder_name[:2] != "gn":
                norm_layer = nn.BatchNorm2d
                encoder_name = encoder_name[2:]
            encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                norm_layer=norm_layer,
                pretrained_strict=False,
            )
            cut = -2

        self.encoder = nn.Sequential(*(list(encoder.children())[:cut] + [nn.ReLU()]))
        encoder_sizes, idxs = self._register_output_hooks(input_shape=input_shape)
        n_chans = int(encoder_sizes[-1][1])
        middle_conv = nn.Sequential(
            ConvBnRelu(n_chans, n_chans // 2, 3, norm_layer=norm_layer),
            ConvBnRelu(n_chans // 2, n_chans, 3, norm_layer=norm_layer),
        )
        decoder = [middle_conv]
        for k, (idx, hook) in enumerate(zip(idxs[::-1], self.hooks)):
            skip_chans = int(encoder_sizes[idx][1])
            final_div = k != len(idxs) - 1
            decoder.append(
                DecoderBlock(
                    n_chans,
                    skip_chans,
                    hook,
                    final_div=final_div,
                    norm_layer=norm_layer,
                )
            )
            n_chans = n_chans // 2 + skip_chans
            n_chans = n_chans if not final_div else skip_chans
        self.decoder = nn.Sequential(*decoder, PixelShuffleICNR(n_chans, n_chans))
        self.head = nn.Sequential(
            nn.Conv2d(n_chans + input_shape[0], n_chans, 1),
            LastCross(n_chans, norm_layer=norm_layer),
            nn.Conv2d(n_chans, n_classes, 1),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = interpolate(y, x.shape[-2:], mode="nearest")
        y = torch.cat([x, y], dim=1)
        y = self.head(y)
        return y

    def _register_output_hooks(self, input_shape=(3, 224, 224)):
        sizes, modules = get_sizes(self.encoder, input_shape=input_shape)
        mods = []
        idxs = np.where(sizes[:-1, -1] != sizes[1:, -1])[0]

        def _hook(model, input, output):
            return output

        for k in idxs[::-1]:
            out_shape = sizes[k]
            m = modules[k]
            if "downsample" not in m.name:
                mods.append(m)
        self.hooks = Hooks(mods, _hook)

        return sizes, idxs

    def __del__(self):
        if hasattr(self, "hooks"):
            self.hooks.remove()



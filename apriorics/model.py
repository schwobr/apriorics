from torch import nn

class AutoUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
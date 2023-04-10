from torch import nn
from torch import device, cuda

device = device("cuda" if cuda.is_available() else "cpu")


class DynamicPoseModule(nn.Module):

    def __init__(self):
        super(DynamicPoseModule, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(792, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        x = x.squeeze(-1)
        return x


class StartPoseModule(nn.Module):

    def __init__(self):
        super(StartPoseModule, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(99, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        x = x.squeeze(-1)
        return x

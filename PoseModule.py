from torch import nn
from torch import device, cuda

device = device("cuda" if cuda.is_available() else "cpu")


class PoseModule(nn.Module):

    def __init__(self):
        super(PoseModule, self).__init__()
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
        ).to(device)

    def forward(self, x):
        # x = self.model(x)
        for module in self._modules.values():
            x = module(x)
        x = x.squeeze(-1)
        return x

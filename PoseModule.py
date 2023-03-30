from torch import nn


class PoseModule(nn.Module):

    def __len__(self):
        super(PoseModule, self).__init__()
        self.module1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(480, 560),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(560, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.module1(x)

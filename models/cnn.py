import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, h=28, w=28, n_channels=1):
        super(Net, self).__init__()
        self.h = h
        self.w = w
        self.n_channels = n_channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2)),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            # nn.Dropout(),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            # nn.Linear(320, 128),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def x_compat(self, x):
        # print(f"x - type {type(x)}")
        if type(x) is not torch.Tensor:
            x_tensor = torch.from_numpy(x)
        else:
            x_tensor = x
        x_out = x_tensor.clone().reshape(x.shape[0],
                                         self.n_channels,
                                         self.h,
                                         self.w)
        return x_out.float()

    def forward(self, x):
        if x.ndim != 4:
            # print("in compat")
            x = self.x_compat(x)
        x = self.conv_layers(x)
        # print(f"shape: {x.shape}")
        # x = x.view(-1, 320)
        x = x.view(-1, 1600)
        x = self.fc_layers(x)
        return x

import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, img_channels=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            self.block(img_channels, base),
            self.block(base, base*2),
            self.block(base*2, base*4),
            self.block(base*4, base*8),
            nn.Conv2d(base*8, 1, 4, 1, 0)
        )

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x).view(-1)

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            self.block(z_dim, base*16, 4, 1, 0),
            self.block(base*16, base*8, 4, 2, 1),
            self.block(base*8, base*4, 4, 2, 1),
            self.block(base*4, base*2, 4, 2, 1),
            self.block(base*2, base, 4, 2, 1),
            nn.ConvTranspose2d(base, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, z):
        return self.net(z)

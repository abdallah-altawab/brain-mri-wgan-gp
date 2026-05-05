import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_samples(generator, z_dim, device, path, n=25):
    generator.eval()
    z = torch.randn(n, z_dim, 1, 1, device=device)
    with torch.no_grad():
        fake = generator(z)
    fake = (fake + 1) / 2
    grid = make_grid(fake, nrow=5)
    plt.figure(figsize=(5,5))
    plt.axis("off")
    plt.imshow(grid.cpu().permute(1,2,0).squeeze(), cmap="gray")
    plt.savefig(path)
    plt.close()

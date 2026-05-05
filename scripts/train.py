import yaml
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.generator import Generator
from models.critic import Critic
from training.trainer import train_step
from utils.seed import set_seed
from utils.visualization import save_samples
import os
from PIL import Image

# =================================================
# Custom Dataset Class (module-level for pickling)
# =================================================
class SingleFolderDataset(datasets.VisionDataset):
    def __init__(self, folder, transform=None):
        super().__init__(folder, transform=transform)
        self.images = [os.path.join(folder, f) for f in os.listdir(folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

# =================================================
# Main function for Windows multiprocessing safety
# =================================================
def main():
    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    with open("configs/train.yaml") as f:
        cfg = yaml.safe_load(f)

    # -------------------------------------------------
    # Seed & device
    # -------------------------------------------------
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------------------------
    # Dataset & DataLoader  ✅ Updated for single-folder dataset
    # -------------------------------------------------
    dataset_path = "data/brain_mri/images/all"  # your GAN images folder
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = SingleFolderDataset(dataset_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    print(f"Loaded {len(dataset)} images from {dataset_path}")

    # -------------------------------------------------
    # Models
    # -------------------------------------------------
    gen = Generator(cfg["training"]["z_dim"]).to(device)
    crit = Critic().to(device)

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------
    opt_g = torch.optim.Adam(
        gen.parameters(),
        lr=cfg["training"]["lr_generator"],
        betas=(0.0, 0.9)
    )

    opt_c = torch.optim.Adam(
        crit.parameters(),
        lr=cfg["training"]["lr_critic"],
        betas=(0.0, 0.9)
    )

    # -------------------------------------------------
    # Create output directories
    # -------------------------------------------------
    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(cfg["training"]["epochs"]):
        g_loss, c_loss = train_step(
            gen,
            crit,
            loader,
            opt_g,
            opt_c,
            {
                "critic_steps": cfg["training"]["critic_steps"],
                "lambda_gp": cfg["training"]["lambda_gp"],
                "z_dim": cfg["training"]["z_dim"],
            },
            device
        )

        print(f"Epoch [{epoch}/{cfg['training']['epochs']}] | "
              f"G Loss: {g_loss:.4f} | C Loss: {c_loss:.4f}")

        # Save sample images
        if epoch % cfg["logging"]["sample_every"] == 0:
            save_samples(
                gen,
                cfg["training"]["z_dim"],
                device,
                f"outputs/images/epoch_{epoch}.png"
            )

        # Save checkpoints
        if epoch % cfg["logging"]["save_every"] == 0:
            torch.save(
                gen.state_dict(),
                f"checkpoints/gen_{epoch}.pt"
            )

# =================================================
# Run main with Windows multiprocessing safety
# =================================================
if __name__ == "__main__":
    main()

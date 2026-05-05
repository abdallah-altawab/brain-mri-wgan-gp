import torch
from models.generator import Generator
from utils.visualization import save_samples

device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator().to(device)
gen.load_state_dict(torch.load("checkpoints/gen_latest.pt", map_location=device))

save_samples(gen, 100, device, "outputs/images/generated.png")

import torch
from training.losses import gradient_penalty

def train_step(gen, crit, loader, opt_g, opt_c, cfg, device):
    gen.train()
    crit.train()

    for real, _ in loader:
        real = real.to(device)
        batch = real.size(0)

        # Critic updates
        for _ in range(cfg["critic_steps"]):
            z = torch.randn(batch, cfg["z_dim"], 1, 1, device=device)
            fake = gen(z)

            gp = gradient_penalty(crit, real, fake, device)
            loss_c = -(crit(real).mean() - crit(fake.detach()).mean()) + cfg["lambda_gp"] * gp

            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        # Generator update
        z = torch.randn(batch, cfg["z_dim"], 1, 1, device=device)
        fake = gen(z)
        loss_g = -crit(fake).mean()

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    return loss_g.item(), loss_c.item()

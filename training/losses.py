import torch

def gradient_penalty(critic, real, fake, device):
    batch = real.size(0)
    epsilon = torch.rand(batch, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    score = critic(interpolated)
    grad = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True
    )[0]

    grad = grad.view(batch, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

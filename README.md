# 🧠 Brain MRI Synthesis — WGAN-GP

!\[Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
!\[PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
!\[Model](https://img.shields.io/badge/Model-WGAN--GP-purple)
!\[Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-red)
!\[Status](https://img.shields.io/badge/Status-Complete-brightgreen)
!\[License](https://img.shields.io/badge/License-MIT-lightgrey)

I built this project to synthesize realistic grayscale brain MRI images using a
**Wasserstein GAN with Gradient Penalty (WGAN-GP)**. The core motivation was
the data scarcity problem in medical imaging — acquiring labeled MRI scans is
expensive, slow, and heavily regulated. A well-trained generative model solves
this by learning the real data distribution and producing an unlimited number of
novel, statistically consistent samples without touching any patient records.

\---

## Table of Contents

1. [Motivation](#motivation)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Training](#training)
6. [Inference](#inference)
7. [Architecture](#architecture)
8. [Configuration](#configuration)
9. [Results](#results)
10. [Known Limitations](#known-limitations)
11. [References](#references)

\---

## Motivation

Standard GANs are notoriously difficult to train. The discriminator saturates
early, gradients to the generator vanish, and training collapses into
repetitive outputs — a failure mode called mode collapse. I chose **WGAN-GP**
specifically because:

* It replaces Jensen-Shannon divergence with the **Wasserstein-1 (Earth
Mover's) distance** — a metric that remains meaningful and provides useful
gradients even when real and fake distributions have no overlap.
* The **Gradient Penalty** enforces the Lipschitz constraint required by
Wasserstein theory without weight clipping, which caused gradient explosion
in the original WGAN. The result is a more stable, better-conditioned Critic.
* Loss curves are **interpretable** — the Wasserstein estimate actually
correlates with visual quality, unlike standard GAN loss values which tell
you nothing about output quality.

For medical imaging specifically, generating synthetic MRI data opens pathways
for data augmentation, class balancing in pathology detection models, and
privacy-preserving dataset sharing.

\---

## How It Works

```
  z \~ N(0, I)  →  \[ Generator ]  →  Fake MRI (128×128)
                                            │
  Real MRI (64×64) ───────────────→  \[ Critic ]  →  Wasserstein Score
                          ↑
               \[ Gradient Penalty ]
            x̂ = ε·real + (1-ε)·fake
            GP = E\[(||∇C(x̂)||₂ - 1)²]
```

The training loop at each batch:

1. **Load batch** — preprocessed grayscale MRI images, normalized to `\[-1, 1]`
2. **Sample noise** — `z \~ N(0,1)` shaped `\[B, z\_dim, 1, 1]`
3. **Generate** — the Generator maps noise through 6 transposed convolution
blocks to a 128×128 fake MRI image
4. **Score** — real and fake images pass through the Critic independently;
it outputs an unbounded scalar score per image (no sigmoid)
5. **Gradient Penalty** — computed on images interpolated between real and
fake; penalizes gradient norms deviating from 1
6. **Update Critic** `critic\_steps` times:
`L\_C = -(C(real) - C(fake)) + λ·GP`
7. **Update Generator** once:
`L\_G = -C(G(z))`

The Critic is updated multiple times before each Generator step to maintain a
tight Wasserstein estimate before the Generator receives a gradient signal.

\---

## Project Structure

```
brain-mri-wgan-gp/
│
├── configs/
│   └── train.yaml                  # All hyperparameters — edit here, not in code
│
├── models/
│   ├── \_\_init\_\_.py
│   ├── generator.py                # ConvTranspose2d stack: z → 128×128 MRI
│   └── critic.py                   # Conv2d stack: image → scalar Wasserstein score
│
├── training/
│   ├── \_\_init\_\_.py
│   ├── losses.py                   # Gradient penalty — differentiable Lipschitz enforcement
│   └── trainer.py                  # Core adversarial training step
│
├── scripts/
│   ├── \_\_init\_\_.py
│   ├── train.py                    # Entry point: dataset, models, loop, checkpointing
│   └── generate.py                 # Inference: load checkpoint → save sample grid
│
├── utils/
│   ├── \_\_init\_\_.py
│   ├── seed.py                     # Global reproducibility seeding
│   └── visualization.py            # save\_samples utility
│
├── data/
│   └── brain\_mri/images/all/       # Brain MRI images (not tracked in git)
│
├── outputs/                        # Generated images (created at runtime)
├── requirements.txt
├── generator\_final.pt              # Final trained generator weights
├── generated\_100\_images.png        # 100-image output grid
├── random\_samples.png              # Samples from random latent vectors
└── training\_losses.png             # Generator and Critic loss curves
```

\---

## Installation

### Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.12 with CUDA support (recommended)
* torchvision, PyYAML, Pillow, matplotlib

```bash
git clone https://github.com/your-username/brain-mri-wgan-gp.git
cd brain-mri-wgan-gp
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Verify GPU availability before training:

```python
import torch
print(torch.cuda.is\_available())      # Expected: True
print(torch.cuda.get\_device\_name(0))  # Your GPU
```

\---

## Training

Place your MRI images (`.png`, `.jpg`, or `.jpeg`) into:

```
data/brain\_mri/images/all/
```

Then run:

```bash
python scripts/train.py
```

The script reads all settings from `configs/train.yaml`, scans the data folder,
instantiates both models, runs the full training loop, and saves outputs
automatically. No command-line arguments are needed.

Progress is printed per epoch:

```
Epoch \[10/300] | G Loss: -1.2341 | C Loss: -8.4521
```

Samples are saved to `outputs/images/` every `sample\_every` epochs.
Generator checkpoints are saved to `checkpoints/` every `save\_every` epochs.

\---

## Inference

```bash
python scripts/generate.py
```

Loads `generator\_final.pt` and writes a grid of 100 generated MRI images to
`outputs/images/generated.png`. The checkpoint path is currently hardcoded —
edit `generate.py` directly if using a different checkpoint file.

\---

## Architecture

### Generator (`models/generator.py`)

Maps a latent noise vector to a 128×128 grayscale image through progressive
2× spatial upsampling at each block.
Each block: `ConvTranspose2d → BatchNorm2d → ReLU(inplace=True)`.

|Layer|Channels (In → Out)|Spatial Out|Activation|
|-|-|-|-|
|ConvTranspose2d (k=4, s=1, p=0)|z\_dim → 1024|4×4|BatchNorm + ReLU|
|ConvTranspose2d (k=4, s=2, p=1)|1024 → 512|8×8|BatchNorm + ReLU|
|ConvTranspose2d (k=4, s=2, p=1)|512 → 256|16×16|BatchNorm + ReLU|
|ConvTranspose2d (k=4, s=2, p=1)|256 → 128|32×32|BatchNorm + ReLU|
|ConvTranspose2d (k=4, s=2, p=1)|128 → 64|64×64|BatchNorm + ReLU|
|ConvTranspose2d (k=4, s=2, p=1)|64 → 1|**128×128**|**Tanh**|

* `bias=False` in all blocks — BatchNorm subsumes the bias term
* Tanh output maps to `\[-1, 1]`, matching the normalized input range exactly

### Critic (`models/critic.py`)

Progressively downsamples the input image to a single scalar Wasserstein score.
No sigmoid — the output is unbounded by design.
Each block: `Conv2d → InstanceNorm2d(affine=True) → LeakyReLU(0.2, inplace=True)`.

|Layer|Channels (In → Out)|Spatial Out|Activation|
|-|-|-|-|
|Conv2d (k=4, s=2, p=1)|1 → 64|32×32|InstanceNorm + LeakyReLU(0.2)|
|Conv2d (k=4, s=2, p=1)|64 → 128|16×16|InstanceNorm + LeakyReLU(0.2)|
|Conv2d (k=4, s=2, p=1)|128 → 256|8×8|InstanceNorm + LeakyReLU(0.2)|
|Conv2d (k=4, s=2, p=1)|256 → 512|4×4|InstanceNorm + LeakyReLU(0.2)|
|Conv2d (k=4, s=1, p=0)|512 → 1|**1×1**|None|
|`.view(-1)`|—|**scalar**|—|

I used **InstanceNorm** (not BatchNorm) throughout the Critic. BatchNorm couples
sample scores across the batch dimension, violating the i.i.d. assumption
required for a reliable Wasserstein estimate. InstanceNorm normalizes each sample
independently.

> The final Conv2d produces a `1×1` map only when input spatial size is
> `4×4`, which occurs only when `image\_size = 64`. This value \*\*must not be
> changed\*\* without modifying the Critic architecture.

\---

## Configuration

All settings live in `configs/train.yaml`. Nothing is hardcoded in any source
file — every hyperparameter is read from this config at runtime.

|Parameter|Recommended|Description|
|-|-|-|
|`training.z\_dim`|`100`|Latent vector dimensionality|
|`training.batch\_size`|`64`|Images per gradient step|
|`training.epochs`|`300`|Total training epochs|
|`training.critic\_steps`|`5`|Critic updates per Generator step|
|`training.lambda\_gp`|`10`|Gradient penalty weight|
|`training.lr\_generator`|`1e-4`|Generator Adam learning rate|
|`training.lr\_critic`|`1e-4`|Critic Adam learning rate|
|`training.num\_workers`|`2`|DataLoader worker processes|
|`training.seed`|`42`|Global reproducibility seed|
|`dataset.image\_size`|`64`|Resize target — **must be 64**|
|`logging.sample\_every`|`10`|Save sample grid every N epochs|
|`logging.save\_every`|`25`|Save checkpoint every N epochs|

Both optimizers use `Adam(betas=(0.0, 0.9))`.
Setting `β₁=0` disables momentum in the Critic, preventing oscillation caused
by stale gradients accumulated across repeated Critic re-optimization steps —
a standard recommendation for WGAN-GP (Gulrajani et al., 2017).

\---

## Results

|Metric|Detail|
|-|-|
|Output resolution|128×128 grayscale|
|Training stability|Stable convergence — Wasserstein loss tracks quality|
|Sample diversity|Visually varied outputs across latent samples|
|Convergence|\~200–300 epochs depending on dataset size|

Committed outputs:

|File|Contents|
|-|-|
|`generated\_100\_images.png`|Grid of 100 synthesized MRI images|
|`random\_samples.png`|Samples from random points in latent space|
|`training\_losses.png`|Generator and Critic loss curves over training|

\---

## Known Limitations

* **Fixed output resolution** — the architecture produces 128×128 only;
changing resolution requires adding/removing blocks in both models
* **Critic not checkpointed** — only `gen.state\_dict()` is saved; resuming
training from a checkpoint requires critic warm-up before stable dynamics
* **No FID metric** — image quality is currently evaluated qualitatively only
* **Unconditional generation** — no class conditioning; all outputs are sampled
from a single learned distribution
* **Last-batch loss logging** — `train\_step` returns the final batch's loss,
not the epoch average; loss values can be noisy

\---

## References

|Paper|Contribution|
|-|-|
|Arjovsky et al. (2017). *Wasserstein GAN*. ICML.|Wasserstein objective, Critic formulation|
|Gulrajani et al. (2017). *Improved Training of WGANs*. NeurIPS.|Gradient penalty, λ=10, critic\_steps=5|
|Radford et al. (2015). *Unsupervised Representation Learning with DCGANs*. ICLR.|ConvTranspose2d generator template|
|Ulyanov et al. (2016). *Instance Normalization*. arXiv.|InstanceNorm rationale|




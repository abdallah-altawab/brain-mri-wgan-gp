# Software Engineering Report

## Brain MRI Synthesis — WGAN-GP

|Field|Value|
|-|-|
|**Project**|brain-mri-wgan-gp|
|**Version**|1.0.0|
|**Classification**|Technical Engineering Documentation|
|**Framework**|PyTorch 2.x / Python 3.8+|
|**Domain**|Generative Deep Learning — Medical Imaging|
|**Status**|Complete|

\---

## Table of Contents

1. [Abstract](#1-abstract)
2. [System Overview](#2-system-overview)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Module Breakdown](#5-module-breakdown)
6. [Dependency Graph](#6-dependency-graph)
7. [Data Flow and Tensor Shapes](#7-data-flow-and-tensor-shapes)
8. [Training Algorithm](#8-training-algorithm)
9. [Model Architecture Details](#9-model-architecture-details)
10. [Design Decisions](#10-design-decisions)
11. [Configuration System](#11-configuration-system)
12. [Performance Characteristics](#12-performance-characteristics)
13. [Error Catalog](#13-error-catalog)
14. [Extensibility](#14-extensibility)
15. [Known Limitations](#15-known-limitations)
16. [Glossary](#16-glossary)
17. [References](#17-references)

\---

## 1\. Abstract

I designed and implemented a generative system for synthesizing grayscale brain
MRI images using Wasserstein GAN with Gradient Penalty (WGAN-GP). The system
addresses the chronic data scarcity problem in medical deep learning by learning
a generative model of the real MRI distribution — enabling unlimited synthetic
sample production without storing or re-identifying patient data.

The codebase is a modular PyTorch pipeline organized across six primary source
modules: model definition, loss computation, training orchestration, dataset
handling, visualization, and inference. The Critic estimates the Wasserstein-1
distance between real and generated distributions; the Generator maps 100-
dimensional Gaussian noise to 128×128 single-channel MRI images.

\---

## 2\. System Overview

### 2.1 Problem Statement

Brain MRI acquisition is expensive, time-consuming, and subject to strict
patient privacy regulations. Models trained for segmentation, classification,
or anomaly detection suffer directly from this — chronic data scarcity and class
imbalance degrade generalization. I built this generative system to produce
statistically consistent synthetic MRI samples that can augment training sets
without exposing any real patient records.

### 2.2 Why WGAN-GP

|Problem with Standard GAN|WGAN-GP Solution|
|-|-|
|Discriminator saturates early → vanishing gradients|Wasserstein distance provides non-saturating gradients|
|JS divergence undefined for non-overlapping distributions|W-1 distance is always defined and meaningful|
|Weight clipping (original WGAN) → gradient explosion|Gradient penalty enforces Lipschitz constraint softly|
|Loss values uncorrelated with image quality|Wasserstein estimate correlates with visual quality|

### 2.3 Project File Inventory

|File|Lines|Role|
|-|-|-|
|`models/generator.py`|22|6-block ConvTranspose2d generator|
|`models/critic.py`|21|5-layer Conv2d critic, scalar output|
|`training/losses.py`|19|Gradient penalty via `torch.autograd.grad`|
|`training/trainer.py`|34|Adversarial training step|
|`scripts/train.py`|95|Full orchestration: data, models, loop|
|`scripts/generate.py`|10|Inference: load checkpoint, save images|

\---

## 3\. Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                    WGAN-GP TRAINING GRAPH                       ║
╚══════════════════════════════════════════════════════════════════╝

  z \~ N(0,I) \[B, 100, 1, 1]          Real MRI \[B, 1, 64, 64]
          │                                       │
          ▼                                       │
  ╔═══════════════════╗                           │
  ║    GENERATOR      ║                           │
  ║                   ║                           │
  ║  ConvT → 4×4      ║                           │
  ║  ConvT → 8×8      ║                           │
  ║  ConvT → 16×16    ║                           │
  ║  ConvT → 32×32    ║                           │
  ║  ConvT → 64×64    ║                           │
  ║  ConvT → 128×128  ║                           │
  ║  Tanh             ║                           │
  ╚════════╤══════════╝                           │
           │ fake \[B, 1, 128, 128]                │
           │                                      │
           ├──────────────────────────────────────┤
           │                                      │
           │   ╔══════════════════════════════╗   │
           │   ║    GRADIENT PENALTY          ║   │
           │   ║  ε \~ Uniform(0,1)            ║   │
           │   ║  x̂ = ε·real + (1-ε)·fake    ║   │
           │   ║  x̂.requires\_grad = True     ║   │
           │   ║  score = C(x̂)               ║   │
           │   ║  ∇ = autograd.grad(score,x̂  ║   │
           │   ║        create\_graph=True)    ║   │
           │   ║  GP = E\[(||∇||₂ - 1)²]      ║   │
           │   ╚══════════════╤═══════════════╝   │
           │                  │                   │
           ▼                  ▼                   ▼
  ╔══════════════════════════════════════════════════╗
  ║                   CRITIC                        ║
  ║  Conv(1→64)   + InstanceNorm + LeakyReLU(0.2)  ║
  ║  Conv(64→128) + InstanceNorm + LeakyReLU(0.2)  ║
  ║  Conv(128→256)+ InstanceNorm + LeakyReLU(0.2)  ║
  ║  Conv(256→512)+ InstanceNorm + LeakyReLU(0.2)  ║
  ║  Conv(512→1)  → 1×1 spatial → .view(-1)        ║
  ╚══════╤══════════════════════════╤═══════════════╝
         │                          │
      C(fake)                   C(real)
         └────────────┬─────────────┘
                      │
   L\_C = -(C(real).mean() - C(fake.detach()).mean()) + λ·GP
   L\_G = -C(fake).mean()

   ┌─────────────────────────────────────────────────┐
   │   Both: Adam(lr=1e-4, betas=(0.0, 0.9))         │
   │   β₁=0 prevents critic momentum accumulation    │
   └─────────────────────────────────────────────────┘
```

\---

## 4\. Mathematical Formulation

### 4.1 Wasserstein-1 Distance

By the Kantorovich-Rubinstein duality theorem, the Wasserstein-1 distance
between real distribution P\_r and generated distribution P\_g is:

```
W(P\_r, P\_g) = sup\_{||f||\_L ≤ 1}  E\_{x\~P\_r}\[f(x)] - E\_{x\~P\_g}\[f(x)]
```

The Critic C approximates the optimal 1-Lipschitz function f. Training C to
maximize `E\[C(real)] - E\[C(fake)]` gives an estimate of W(P\_r, P\_g).

### 4.2 Critic Objective

```
L\_C = -( E\_{x\~P\_r}\[C(x)] - E\_{z\~P\_z}\[C(G(z))] ) + λ · GP
```

|Term|Role|
|-|-|
|`-E\[C(real)]`|Push Critic scores on real images upward|
|`+E\[C(fake)]`|Push Critic scores on fake images downward|
|`λ · GP`|Penalize violations of the 1-Lipschitz constraint|

`fake.detach()` is used in `L\_C` to block gradients from flowing into the
Generator during the Critic update step.

### 4.3 Gradient Penalty

```
x̂  = ε · x\_real + (1-ε) · x\_fake         ε \~ Uniform(0,1)
GP = E\[ ( ||∇\_{x̂} C(x̂)||₂ - 1 )² ]
```

Implementation in `training/losses.py`:

```python
epsilon = torch.rand(batch, 1, 1, 1, device=device)
interpolated = epsilon \* real + (1 - epsilon) \* fake
interpolated.requires\_grad\_(True)

score = critic(interpolated)
grad  = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            grad\_outputs=torch.ones\_like(score),
            create\_graph=True,     # gradient of GP must itself be differentiable
            retain\_graph=True
        )\[0]

grad = grad.view(batch, -1)
return ((grad.norm(2, dim=1) - 1) \*\* 2).mean()
```

`create\_graph=True` is mandatory. The gradient penalty is a term inside `L\_C`,
and `L\_C.backward()` must differentiate through it. Without `create\_graph=True`,
that second-order differentiation raises a `RuntimeError`.

### 4.4 Generator Objective

```
L\_G = -E\_{z\~P\_z}\[C(G(z))]
```

The Generator is not detached here — `L\_G.backward()` propagates through the
Critic and back into the Generator's parameters.

\---

## 5\. Module Breakdown

### `models/generator.py` — Generator

**Input:** `z: Tensor\[B, z\_dim, 1, 1]`
**Output:** `Tensor\[B, 1, 128, 128]` in range `\[-1, 1]`

Six-block `ConvTranspose2d` stack using `BatchNorm2d + ReLU` per block,
with a final `ConvTranspose2d + Tanh`. All blocks set `bias=False` since
BatchNorm subsumes the bias term. Spatial resolution doubles at each of the
last five blocks via `stride=2`.

### `models/critic.py` — Critic

**Input:** `Tensor\[B, 1, H, W]`
**Output:** `Tensor\[B]` — one unbounded scalar score per sample

Four-block `Conv2d` stack using `InstanceNorm2d(affine=True) + LeakyReLU(0.2)`
per block, followed by `Conv2d(512, 1, 4, 1, 0)` and `.view(-1)`. No sigmoid.
The 1×1 spatial output from the final layer is valid only when input `H=W=64`.

### `training/losses.py` — Gradient Penalty

**Inputs:** `critic`, `real: Tensor\[B,C,H,W]`, `fake: Tensor\[B,C,H,W]`, `device`
**Output:** scalar `Tensor` — mean squared gradient norm deviation

Pure function with no side effects. Computes the interpolation, runs a Critic
forward pass with `requires\_grad=True`, and uses `torch.autograd.grad` with
`create\_graph=True` to produce a differentiable penalty.

### `training/trainer.py` — Training Step

**Inputs:** `gen`, `crit`, `loader`, `opt\_g`, `opt\_c`, `cfg`, `device`
**Returns:** `(g\_loss: float, c\_loss: float)` — last-batch values

Runs `critic\_steps` Critic updates followed by one Generator update per batch.
Calls `gen.train()` and `crit.train()` on entry to activate training-mode
normalization layers.

### `scripts/train.py` — Orchestration

Entry point. Reads `configs/train.yaml`, builds `SingleFolderDataset`
(custom `VisionDataset` subclass), instantiates both models, configures both
Adam optimizers, runs the epoch loop, and handles all I/O — sample saving
and checkpointing.

### `scripts/generate.py` — Inference

Loads `generator\_final.pt` using `torch.load(..., map\_location=device)`,
instantiates `Generator()` with default arguments, and calls `save\_samples`.
No CLI arguments — checkpoint path is hardcoded.

\---

## 6\. Dependency Graph

```
scripts/train.py  (entry point)
│
├── configs/train.yaml
├── models/generator.py          └── torch.nn
├── models/critic.py             └── torch.nn
│
├── training/trainer.py
│   ├── torch
│   └── training/losses.py       └── torch / torch.autograd.grad
│
├── utils/seed.py                └── (torch, random, numpy) 
├── utils/visualization.py       └── (torch, matplotlib / torchvision) 
│
└── torch / torchvision / PIL / os / yaml

scripts/generate.py
├── models/generator.py
├── utils/visualization.py
└── torch
```

\---

## 7\. Data Flow and Tensor Shapes

### 7.1 Full Pipeline Tensor Trace

|Stage|Tensor Shape|Range|Notes|
|-|-|-|-|
|Raw image on disk|`\[H, W]` PIL|`\[0, 255]`|`.jpg` / `.png` / `.jpeg`|
|After `Grayscale`|`\[H, W]` PIL|`\[0, 255]`|1 channel|
|After `Resize`|`\[64, 64]` PIL|`\[0, 255]`|Must be 64|
|After `ToTensor`|`\[1, 64, 64]`|`\[0.0, 1.0]`||
|After `Normalize(\[0.5],\[0.5])`|`\[1, 64, 64]`|`\[-1.0, 1.0]`|Matches Tanh output|
|DataLoader batch|`\[B, 1, 64, 64]`|`\[-1.0, 1.0]`||
|Latent vector `z`|`\[B, 100, 1, 1]`|`(-∞, ∞)`|`torch.randn`|
|Generator output|`\[B, 1, 128, 128]`|`\[-1.0, 1.0]`||
|Interpolated sample|`\[B, 1, 64, 64]`|`\[-1.0, 1.0]`|For GP|
|Gradient (flattened)|`\[B, 64×64]`|`(-∞, ∞)`|`grad.view(B, -1)`|
|Critic score|`\[B]`|`(-∞, ∞)`|Wasserstein estimate|
|GP term|`\[]` scalar|`\[0, ∞)`||
|`L\_C`, `L\_G`|`\[]` scalar|`(-∞, ∞)`||

### 7.2 Latent Vector Convention

```python
z = torch.randn(batch, cfg\["z\_dim"], 1, 1, device=device)
```

The trailing `(1, 1)` spatial dimensions are required by the Generator's
first `ConvTranspose2d(z\_dim, 1024, kernel\_size=4, stride=1, padding=0)`.
A flat `\[B, z\_dim]` vector would cause a dimension mismatch at this layer.

### 7.3 Epsilon Broadcasting in GP

```python
epsilon = torch.rand(batch, 1, 1, 1, device=device)
interpolated = epsilon \* real + (1 - epsilon) \* fake
```

`epsilon` must be shaped `\[B, 1, 1, 1]` to broadcast correctly across all
spatial positions and channels. A common error is using `torch.rand(batch)`,
which broadcasts incorrectly across spatial dimensions — this corrupts the
interpolation without raising an error.

\---

## 8\. Training Algorithm

### Pseudocode

```
FOR epoch = 1 to epochs:
  FOR (x\_real, \_) in DataLoader:

    ── CRITIC LOOP (×critic\_steps) ─────────────────────────
    FOR t = 1 to critic\_steps:
      z      ← randn(B, z\_dim, 1, 1)
      fake   ← G(z)
      gp     ← gradient\_penalty(C, x\_real, fake)
      L\_C    ← -(C(x\_real).mean() - C(fake.detach()).mean()) + λ·gp
      opt\_C:  zero\_grad → L\_C.backward() → step

    ── GENERATOR UPDATE (×1) ───────────────────────────────
    z      ← randn(B, z\_dim, 1, 1)
    fake   ← G(z)
    L\_G    ← -C(fake).mean()
    opt\_G:  zero\_grad → L\_G.backward() → step

  ── END BATCH LOOP ──────────────────────────────────────────
  IF epoch % sample\_every == 0: save\_samples(...)
  IF epoch % save\_every    == 0: torch.save(gen.state\_dict(), ...)
```

### Compute Cost Per Batch (`critic\_steps=5`)

|Operation|Count|Relative Cost|
|-|-|-|
|Generator forward|6|1× each|
|Critic forward|16|\~0.7× each|
|Critic backward with GP (`create\_graph`)|5|\~3× standard backward|
|Generator backward|1|\~1×|

The `create\_graph=True` backward is the dominant per-step cost. At
`critic\_steps=5`, the Critic accounts for approximately 80% of wall-clock
time per batch. The Wasserstein estimate quality improves with more critic
steps, but compute cost scales linearly.

\---

## 9\. Model Architecture Details

### 9.1 Generator — Full Layer Table

All intermediate blocks: `ConvTranspose2d(bias=False) → BatchNorm2d → ReLU(inplace=True)`

|#|Kernel/Stride/Pad|Ch In → Out|Spatial Out|Params (approx)|
|-|-|-|-|-|
|1|k=4, s=1, p=0|z\_dim → 1024|1×1 → 4×4|1.64M|
|2|k=4, s=2, p=1|1024 → 512|4×4 → 8×8|8.39M|
|3|k=4, s=2, p=1|512 → 256|8×8 → 16×16|2.10M|
|4|k=4, s=2, p=1|256 → 128|16×16 → 32×32|524K|
|5|k=4, s=2, p=1|128 → 64|32×32 → 64×64|131K|
|6|k=4, s=2, p=1|64 → 1|64×64 → 128×128|1K + Tanh|

**Total: \~12.8M parameters (including BatchNorm)**

Spatial formula: `H\_out = (H\_in - 1) × stride - 2×padding + kernel`

### 9.2 Critic — Full Layer Table

All intermediate blocks: `Conv2d → InstanceNorm2d(affine=True) → LeakyReLU(0.2, inplace=True)`

|#|Kernel/Stride/Pad|Ch In → Out|Spatial Out|Notes|
|-|-|-|-|-|
|1|k=4, s=2, p=1|1 → 64|64×64 → 32×32||
|2|k=4, s=2, p=1|64 → 128|32×32 → 16×16||
|3|k=4, s=2, p=1|128 → 256|16×16 → 8×8||
|4|k=4, s=2, p=1|256 → 512|8×8 → 4×4||
|5|k=4, s=1, p=0|512 → 1|4×4 → **1×1**|No activation|
|—|`.view(-1)`|—|**scalar**|One score per sample|

**Spatial alignment requirement:**

```
64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1  (image\_size=64)
128×128 → 64×64 → 32×32 → 16×16 → 8×8 → 5×5  (image\_size=128)
```

`view(-1)` on a `\[B, 1, 5, 5]` output produces `\[B·25]`, silently corrupting
the loss computation without raising an error.

\---

## 10\. Design Decisions

### `fake.detach()` in Critic Loss

```python
loss\_c = -(crit(real).mean() - crit(fake.detach()).mean()) + λ\*gp
```

Without `.detach()`, `loss\_c.backward()` would propagate gradients into the
Generator during the Critic update. This breaks the adversarial separation:
the Generator would receive gradient updates from both `L\_C` and `L\_G`,
corrupting the training balance. `.detach()` severs this connection cleanly.

### InstanceNorm in Critic, BatchNorm in Generator

BatchNorm in the Critic couples the score assigned to one sample with the
statistics of all other samples in the batch — the Critic's judgment becomes
batch-composition-dependent. This violates the i.i.d. assumption that
underpins the Wasserstein estimate. InstanceNorm normalizes per-sample,
per-channel, keeping scores statistically independent.

In the Generator, BatchNorm is standard and appropriate — it stabilizes feature
distributions during upsampling where cross-sample statistics are beneficial.

### `create\_graph=True` in Gradient Penalty

The gradient penalty is computed inside `L\_C`, which is then passed to
`L\_C.backward()`. That backward call must differentiate through the GP term,
meaning it needs the gradient of a gradient — a second-order operation.
`create\_graph=True` retains the computational graph of the first-order
gradient so the second-order derivative can be computed. Omitting it raises
`RuntimeError: element 0 of tensors does not require grad`.

### `Adam(β₁=0.0, β₂=0.9)`

Standard Adam accumulates first-moment momentum from prior gradients.
In adversarial training, the Critic is repeatedly re-optimized against a
continuously changing Generator distribution. Stale momentum from earlier
Critic steps causes oscillation. Setting `β₁=0` disables momentum entirely,
a deliberate WGAN-GP convention from Gulrajani et al. (2017).

\---

## 11\. Configuration System

All hyperparameters are externalized in `configs/train.yaml`. The schema
below is reconstructed from all `cfg\[...]` usages across the source files.

```yaml
training:
  seed:          42       # Global reproducibility seed
  batch\_size:    64       # Images per gradient step
  epochs:        300      # Total training epochs
  z\_dim:         100      # Latent vector size (must match Generator constructor)
  critic\_steps:  5        # Critic updates per Generator step
  lambda\_gp:     10       # Gradient penalty weight
  lr\_generator:  0.0001   # Generator Adam learning rate
  lr\_critic:     0.0001   # Critic Adam learning rate
  num\_workers:   2        # DataLoader worker processes

dataset:
  image\_size:    64       # CRITICAL: must be 64 for Critic architecture

logging:
  sample\_every:  10       # Save sample grid every N epochs
  save\_every:    25       # Save checkpoint every N epochs
```

### Critical Parameter Interactions

```
z\_dim ←→ Generator(z\_dim=...)  ←→  torch.randn(B, z\_dim, 1, 1)
image\_size ←→ transforms.Resize ←→ Critic spatial trace (must be 64)
critic\_steps ←→ inner for loop in train\_step
lambda\_gp ←→ loss\_c = ... + lambda\_gp \* gp
```

Any mismatch between `z\_dim` in config and the instantiated Generator will
cause a `load\_state\_dict` key-size mismatch at inference time. These values
must be versioned together.

\---

## 12\. Performance Characteristics

### GPU Memory Footprint (Approximate, B=64, image\_size=64)

|Component|Memory|
|-|-|
|Generator weights|\~50 MB|
|Critic weights|\~40 MB|
|Adam optimizer states (×2 per model)|\~180 MB total|
|Real batch `\[64, 1, 64, 64]`|\~1 MB|
|Fake batch + computation graph|\~15–25 MB|
|Interpolated batch + `create\_graph` overhead|\~30–50 MB|
|**Estimated total**|**\~4–8 GB**|

### DataLoader Configuration

```python
DataLoader(dataset, batch\_size=..., shuffle=True,
           num\_workers=N, pin\_memory=True)
```

`pin\_memory=True` enables asynchronous CPU→GPU transfer, overlapping data
loading with GPU compute. `num\_workers > 0` enables prefetching; the
`if \_\_name\_\_ == "\_\_main\_\_"` guard in `train.py` is required for
multiprocessing safety on Windows.

### Identified Bottlenecks

|Bottleneck|Cause|Mitigation|
|-|-|-|
|GP second-order backward|`create\_graph=True` — inherent to WGAN-GP|None — required by design|
|High critic-to-generator ratio|`critic\_steps=5` — 5× critic compute per step|Reduce to 3 at cost of looser estimate|
|No mixed precision|No `torch.cuda.amp` usage|Add `autocast` + `GradScaler`|
|No critic checkpoint|Training cannot resume cleanly|Save full checkpoint dict per epoch|

\---

## 13\. Error Catalog

|Error|Location|Cause|Resolution|
|-|-|-|-|
|`RuntimeError: view(-1) shape mismatch`|`critic.py`|`image\_size ≠ 64`; final Conv produces non-1×1 output|Set `dataset.image\_size: 64` in config|
|`RuntimeError: element 0 does not require grad`|`losses.py`|`create\_graph=True` removed from autograd call|Do not modify `losses.py`|
|`RuntimeError: Expected all tensors on same device`|`trainer.py`|`real` or `z` not moved to `device`|Check `real = real.to(device)` and `z = torch.randn(..., device=device)`|
|`KeyError` on config access|`train.py`, `trainer.py`|Missing key in `train.yaml`|Validate config against schema on load|
|`FileNotFoundError: gen\_latest.pt`|`generate.py`|Training saves `gen\_{epoch}.pt`; `gen\_latest.pt` does not exist|Rename desired checkpoint or update path in `generate.py`|
|Silent loss corruption|`training/losses.py`|`epsilon` shaped `\[B]` not `\[B,1,1,1]` — wrong broadcast|Verify epsilon shape is `\[B, 1, 1, 1]`|
|Mode collapse (no error raised)|Training|Generator ignores `z`; produces repetitive outputs|Increase `lambda\_gp`; increase `critic\_steps`|
|Critic loss diverges|Training|GP term overwhelmed by Wasserstein term|Reduce `lambda\_gp`; check `image\_size` alignment|
|`RuntimeError: Expected input batch\_size ≠ 1` with BatchNorm|`generator.py`|BatchNorm undefined at `batch\_size=1`|Use `batch\_size ≥ 2`; or replace BN with InstanceNorm in Generator|

\---

## 14\. Extensibility

### Swap the Dataset

`SingleFolderDataset` in `scripts/train.py` is a self-contained
`VisionDataset` subclass. Replace it with any class that returns
`(image\_tensor: \[1, H, W], label: int)` from `\_\_getitem\_\_`. The rest of the
pipeline is unaffected.

### Change Output Resolution

To target a resolution other than 128×128:

1. Add/remove `ConvTranspose2d` blocks in `Generator`
2. Adjust `Conv2d` blocks in `Critic` so the final layer's input is `4×4`
3. Update `dataset.image\_size` in `train.yaml` to the new input resolution

The spatial alignment constraint is: `image\_size / 2^(num\_critic\_blocks) = 4`.
For 4 downsampling blocks: `image\_size = 64`.

### Add Full Training Resumption

Currently only `gen.state\_dict()` is saved. For clean resumption, save and
restore the full training state:

```python
torch.save({
    'epoch':             epoch,
    'gen\_state\_dict':    gen.state\_dict(),
    'crit\_state\_dict':   crit.state\_dict(),
    'opt\_g\_state\_dict':  opt\_g.state\_dict(),
    'opt\_c\_state\_dict':  opt\_c.state\_dict(),
}, f"checkpoints/checkpoint\_{epoch}.pt")
```

### Add Quantitative Evaluation (FID)

```bash
pip install pytorch-fid
python -m pytorch\_fid path/to/real\_images/ path/to/generated/ --device cuda
```

FID measures distributional distance between real and generated images using
Inception features. It is the standard metric for GAN quality evaluation and
should be computed every `N` epochs alongside visual samples.

### Enable Mixed Precision Training

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    fake  = gen(z)
    gp    = gradient\_penalty(crit, real, fake, device)
    loss\_c = ...
scaler.scale(loss\_c).backward()
scaler.step(opt\_c)
scaler.update()
```

Test `create\_graph=True` interaction with AMP on the target hardware before
deploying — second-order gradient computation with mixed precision requires
careful dtype handling.

\---

## 15\. Known Limitations

|Limitation|Detail|
|-|-|
|Fixed output resolution|Architecture is hardcoded to 128×128; changing it requires modifying both models|
|Critic not checkpointed|Only `gen.state\_dict()` saved; resumed training requires critic warm-up|
|Last-batch loss only|`train\_step` returns final batch loss, not epoch average — values can be noisy|
|No FID / quantitative metric|Quality assessed qualitatively only|
|Hardcoded paths in `generate.py`|No CLI — paths must be edited directly in source|
|`image\_size` not validated at runtime|Wrong value silently corrupts Critic output and loss|
|Unconditional generation|No class conditioning or pathology-specific control|
|Scanner bias|Model learns distribution of training data exactly — single-scanner datasets produce scanner-specific artifacts|

\---

## 16\. Glossary

|Term|Definition|
|-|-|
|**WGAN-GP**|Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017)|
|**Critic**|WGAN's equivalent of a discriminator; outputs an unbounded real-valued score|
|**Wasserstein-1 Distance**|Earth Mover's distance; minimum cost to transform one distribution into another|
|**Gradient Penalty**|Regularization term penalizing `|
|**Lipschitz Constraint**|Requirement that `|
|**Mode Collapse**|Generator failure where outputs lose diversity; ignores latent noise|
|**InstanceNorm**|Per-sample, per-channel normalization; preserves independence across batch|
|**BatchNorm**|Per-channel normalization across the batch dimension|
|**ConvTranspose2d**|Fractionally-strided convolution; used for spatial upsampling|
|**`create\_graph=True`**|PyTorch flag enabling second-order differentiation through autograd|
|**`fake.detach()`**|Severs tensor from computation graph; blocks Generator gradients during Critic update|
|**FID**|Fréchet Inception Distance — standard GAN quality metric|
|**Adam (β₁=0)**|Adam with zero momentum; standard for WGAN-GP optimizers|

\---

## 17\. References

|Paper|Contribution|
|-|-|
|Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS.|Original GAN formulation|
|Arjovsky et al. (2017). *Wasserstein GAN*. ICML.|Wasserstein distance objective; Critic design|
|Gulrajani et al. (2017). *Improved Training of Wasserstein GANs*. NeurIPS.|Gradient penalty; λ=10, critic\_steps=5 recommendations|
|Radford et al. (2015). *Unsupervised Representation Learning with DCGANs*. ICLR.|ConvTranspose2d generator architecture template|
|Ulyanov et al. (2016). *Instance Normalization*. arXiv.|InstanceNorm rationale for image synthesis networks|




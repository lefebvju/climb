# CLIMB: Centroid-Based Hierarchical Memory for Online Continual Self-Supervised Learning

> Julien Lefebvre, Stefan Duffner, Mathieu Lefort  
> Université Claude Bernard Lyon 1 / INSA Lyon, CNRS, LIRIS  
> Accepted at CoLLAs 2026

## Overview

**CLIMB** (*Continual Learning with Intelligent Memory Bank*) is an Online Continual Self-Supervised Learning (OCSSL) method that learns high-quality representations from a continuous stream of unlabeled data, without knowledge of task boundaries and under strict memory constraints.

CLIMB combines two complementary anti-forgetting mechanisms:

1. **A hierarchical centroid-based memory (STM/LTM)** that maintains a representative subset of the latent space of the stream, prioritizing regions that are hardest to discriminate under the contrastive loss.
2. **EMA-based alignment loss** that constrains the current model's representations to remain aligned with a stable momentum encoder, preventing catastrophic forgetting.

## Method

### Architecture

CLIMB builds on a standard SSL architecture comprising a backbone $f_\theta$ and a projection head $g_\theta$ optimizing a self-supervised pretext task (SimCLR). Two anti-forgetting mechanisms are integrated on top: a hierarchical memory (STM/LTM) for replay, and an EMA model serving as a stable reference to constrain representation drift.

At each step, a stream mini-batch $b_s$ is encoded and subjected to novelty detection to update the memory. A replay batch $b_r$ is sampled from memory and concatenated with the current batch to form $b = b_s \cup b_r$, used for SSL optimization.

### Memory

The memory is divided into two sub-memories:

- **STM (Short-Term Memory):** handles centroid creation and consolidation. When a stream image arrives, its embedding (in projected space) is compared via cosine distance to all existing centroids. If the minimum distance exceeds threshold $\tau$ (95th percentile of the last 1000 observed distances), a new centroid is instantiated. Otherwise, the image is assigned to the nearest centroid and its value is updated via EMA with coefficient $\alpha_\text{stm}$, provided that centroid belongs to the STM. When a centroid reaches $M$ examples, it is promoted to the LTM.
- **LTM (Long-Term Memory):** retains centroids representative of regions well spread across the stream's latent space. When LTM reaches maximum capacity $K$, the two most similar centroids are merged, their examples are pooled, and $M$ images are retained by random selection.

Memory consolidation is triggered when the total number of stored images (STM + LTM) exceeds global threshold $N$. Centroid positions are recalculated with the current encoder, and all STM examples are pruned to one anchor per centroid.

### Training

For each stream mini-batch $b_s$, a replay batch $b_r$ is sampled in equal parts from the LTM and the STM, forming $b = b_s \cup b_r$. The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{SSL}} + \lambda \, \mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{SSL}}$: SimCLR contrastive loss on the full batch $b$.
- $\mathcal{L}_{\text{align}}$: negative cosine similarity between $a_\phi(z_r)$ and $z^{\text{ema}}_r$, where $z_r = g_\theta(f_\theta(b_r^t))$ are the replayed representations, $a_\phi$ is a dedicated projection head, and $z^{\text{ema}}_r = g_{\theta'}(f_{\theta'}(b_r^t))$ are the representations from the EMA encoder $(f_{\theta'}, g_{\theta'})$.

The EMA encoder is updated with momentum $\tau_{\text{ema}}$ after each mini-batch.

## Results

CLIMB consistently outperforms state-of-the-art OCSSL methods on Split CIFAR-100 and Split ImageNet-100, under both regular and irregular task distributions. Performance is evaluated via Final Accuracy (FA) and Continual Accuracy (CA).

**Regular task distribution — Split ImageNet-100:**

| Method | CA (20 tasks) | FA (20 tasks) | CA (50 tasks) | FA (50 tasks) | CA (100 tasks) | FA (100 tasks) |
|---|---|---|---|---|---|---|
| **CLIMB** | **47.46 ± 1.76** | **52.92 ± 1.14** | **46.22 ± 1.27** | **50.34 ± 0.61** | **44.58 ± 1.13** | **50.21 ± 0.40** |
| CLA-E | 45.52 ± 1.22 | 51.03 ± 1.61 | 43.39 ± 0.96 | 46.78 ± 1.30 | 42.20 ± 1.29 | 46.78 ± 1.30 |
| CLA-R | 42.86 ± 1.48 | 49.89 ± 1.12 | 40.41 ± 1.64 | 46.06 ± 0.78 | 39.57 ± 1.43 | 45.81 ± 0.88 |
| Osiris-R | 37.06 ± 1.80 | 42.72 ± 2.00 | 35.64 ± 1.80 | 39.02 ± 0.93 | 35.04 ± 1.71 | 39.44 ± 1.47 |
| MinRed | 35.87 ± 1.99 | 43.34 ± 1.71 | 34.83 ± 2.29 | 42.70 ± 1.11 | 35.07 ± 1.90 | 41.74 ± 0.83 |
| SCALE | 28.70 ± 1.45 | 33.43 ± 0.59 | 29.26 ± 3.64 | 31.74 ± 1.86 | 22.87 ± 3.09 | 28.08 ± 1.86 |

**Regular task distribution — Split CIFAR-100:**

| Method | CA (20 tasks) | FA (20 tasks) | CA (50 tasks) | FA (50 tasks) | CA (100 tasks) | FA (100 tasks) |
|---|---|---|---|---|---|---|
| **CLIMB** | **41.33 ± 0.72** | **44.09 ± 0.30** | **38.68 ± 1.04** | **43.15 ± 0.58** | **38.60 ± 1.22** | **43.37 ± 0.84** |
| CLA-E | 37.59 ± 1.14 | 40.95 ± 0.98 | 36.60 ± 1.38 | 40.65 ± 1.25 | 36.70 ± 1.31 | 41.23 ± 1.27 |
| CLA-R | 39.87 ± 0.88 | 42.89 ± 1.72 | 38.68 ± 1.26 | 41.67 ± 1.85 | 38.84 ± 1.05 | 42.28 ± 2.26 |
| MinRed | 39.34 ± 1.14 | 43.89 ± 1.44 | 38.14 ± 1.08 | 43.62 ± 1.44 | 38.30 ± 1.42 | 43.46 ± 1.48 |
| Osiris-R | 34.13 ± 1.29 | 37.65 ± 0.57 | 31.94 ± 1.08 | 37.19 ± 0.74 | 32.91 ± 1.43 | 35.48 ± 1.51 |
| SCALE | 27.88 ± 1.30 | 31.32 ± 0.40 | 27.40 ± 0.86 | 31.23 ± 0.50 | 27.25 ± 1.15 | 31.14 ± 0.73 |

Full results including irregular task distributions and SimSiam experiments are reported in the paper.

## Installation

### Requirements

- Python ≥ 3.10 (tested with 3.13.3)
- CUDA ≥ 12.0 (tested with 12.6, on NVIDIA V100 32GB)

```bash
pip install -r requirements.txt
```

## Usage

### From a config file (recommended)

Ready-to-use configs with all hyperparameters set:

```bash
# Split CIFAR-100, 20 tasks (λ=2.0, lr=0.3)
python run_from_config.py --config-path configs/config_climb_cifar100.json

# Split ImageNet-100, 20 tasks (λ=1.0, lr=0.1)
python run_from_config.py --config-path configs/config_climb.json \
  --dataset-root /path/to/imagenet --dataset imagenet100
```

### Direct command line

**Split ImageNet-100 (20 tasks):**

```bash
python main.py \
  --strategy climb \
  --model simclr \
  --encoder resnet18 \
  --dataset imagenet100 \
  --dataset-root /path/to/imagenet \
  --num-exps 20 \
  --lr 0.1 \
  --omega 1.0 \
  --mem-size 2500 \
  --stm-size 100 \
  --ltm-max 60 \
  --max-examples-per-centroid 30 \
  --window-size 1000 \
  --novelty-percentile 0.95 \
  --alpha-stm 0.1 \
  --tau-ema 0.999 \
  --mb-passes 3 \
  --tr-mb-size 10 \
  --repl-mb-size 128
```

**Split CIFAR-100 (20 tasks):**

```bash
python main.py \
  --strategy climb \
  --model simclr \
  --encoder resnet18 \
  --dataset cifar100 \
  --num-exps 20 \
  --lr 0.3 \
  --omega 2.0 \
  --mem-size 2500 \
  --stm-size 100 \
  --ltm-max 60 \
  --max-examples-per-centroid 30 \
  --window-size 1000 \
  --novelty-percentile 0.95 \
  --alpha-stm 0.1 \
  --tau-ema 0.999 \
  --mb-passes 3 \
  --tr-mb-size 10 \
  --repl-mb-size 128
```

### Dataset setup

- **Split CIFAR-100**: downloaded automatically by torchvision on first run.
- **Split ImageNet-100**: requires a local copy of [ImageNet](https://www.image-net.org/challenges/LSVRC/index.php). Set `--dataset-root` to the folder containing `train/` and `val/`. 100 classes are drawn randomly using the provided seed.

## Hyperparameters

| Parameter | Symbol (paper) | Default | Description |
|---|---|---|---|
| `--mem-size` | $N$ | 2500 | Global memory capacity (total images before consolidation) |
| `--stm-size` | $L$ | 100 | Max number of centroids in STM |
| `--ltm-max` | $K$ | 60 | Max number of centroids in LTM |
| `--max-examples-per-centroid` | $M$ | 30 | Max images per centroid (also promotion threshold) |
| `--window-size` | $w$ | 1000 | Sliding window size for adaptive novelty threshold $\tau$ |
| `--novelty-percentile` | $p$ | 0.95 | Percentile for novelty threshold $\tau$ |
| `--alpha-stm` | $\alpha_\text{stm}$ | 0.1 | EMA update coefficient for STM centroid embeddings |
| `--tau-ema` | $\tau_\text{ema}$ | 0.999 | EMA momentum for the reference encoder |
| `--omega` | $\lambda$ | 1.0 | Weight of alignment loss $\mathcal{L}_\text{align}$ (Eq. 2). Use **2.0 for CIFAR-100**, 1.0 for ImageNet-100 |
| `--ratio-ltm` | — | 0.5 | Fraction of replay batch $b_r$ sampled from LTM |
| `--lr` | — | 0.1 | Learning rate (SGD). Use **0.3 for CIFAR-100**, 0.1 for ImageNet-100 |
| `--tr-mb-size` | $b_s$ | 10 | Stream mini-batch size |
| `--repl-mb-size` | $b_r$ | 128 | Replay mini-batch size |
| `--mb-passes` | $n_p$ | 3 | Number of passes per mini-batch |

## Supported Baselines

| Method | `--strategy` |
|---|---|
| CLIMB | `climb` |
| CLA-R | `cla_r` |
| CLA-E | `cla_e` |
| CLA-B | `cla_b` |
| CaSSLe | `cassle` |
| CaSSLe-R | `cassle_r` |
| SCALE | `scale` |
| Osiris-R | `osiris_r` |
| MinRed | `minred` |
| Experience Replay | `replay` |
| LUMP | `lump` |
| No strategy | `no_strategy` |

## Supported Datasets

| Dataset | `--dataset` |
|---|---|
| Split CIFAR-100 | `cifar100` |
| Split CIFAR-100 (irregular) | `cifar100-irregular` |
| Split ImageNet-100 | `imagenet100` |
| Split ImageNet-100 (irregular) | `imagenet100-irregular` |
| CLEAR-100 | `clear100` |

## Code Structure

```
src/
├── strategies/
│   ├── climb.py                  # CLIMB (training loop, replay, alignment loss)
│   ├── cla_r.py / cla_e.py / cla_b.py
│   ├── cassle.py / cassle_r.py
│   ├── replay.py / lump.py / minred.py
│   └── abstract_strategy.py
├── standalone_strategies/
│   ├── scale.py
│   └── osiris_r.py
├── buffers/
│   ├── climb_buffer.py           # ClimbBuffer, CentroidMemory, Centroid
│   ├── minred_buffer.py
│   ├── reservoir_buffer.py
│   └── fifo_buffer.py
├── ssl_models/
│   ├── simclr.py / byol.py / simsiam.py
│   ├── barlow_twins.py / mae.py
│   └── abstract_ssl_model.py
├── backbones/
│   ├── custom_resnets.py
│   └── vit.py
├── probing/
│   ├── probing_sklearn.py        # Ridge regression, kNN probes
│   └── probing_pytorch.py        # Linear probe (PyTorch)
├── trainer.py
├── benchmark.py
├── get_datasets.py
└── transforms.py
configs/
├── config_climb_cifar100.json    # CLIMB — Split CIFAR-100 (all hyperparams explicit)
├── config_climb.json             # CLIMB — Split ImageNet-100 (minimal)
└── config_cla_r.json / config_cla_e.json / ...
main.py                           # Entry point
run_from_config.py                # Run from JSON config
```

## Citation

```bibtex
@inproceedings{lefebvre2026climb,
  title     = {{CLIMB}: Centroid-Based Hierarchical Memory for Online Continual Self-Supervised Learning},
  author    = {Lefebvre, Julien and Duffner, Stefan and Lefort, Mathieu},
  booktitle = {Conference on Lifelong Learning Agents (CoLLAs)},
  year      = {2026}
}
```

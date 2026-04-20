# AML 2026 Project — Block-Factorized Discrete Diffusion

Closing the factorization gap in discrete diffusion via locally-coupled reverse models.

## Setup

```bash
pip install -r requirements.txt
```

## Training (MNIST)

```bash
python train_mnist.py --T 4 --epochs 100 --batch_size 128 --lr 3e-4 --seed 42 --block_size 1
```

`--block_size` selects the reverse head: 1 (pixel, default), 2 (1x2), or 4 (2x2).
Samples go to `samples/` every 10 epochs; checkpoints to `checkpoints/`.

## FID Evaluation

```bash
python evaluate_fid.py --checkpoint checkpoints/best.pt --T 4 --n_samples 10000
python -m pytorch_fid fid_stats/real fid_stats/generated
```

## Synthetic experiment (E1)

Controlled dataset: each 2x2 block is i.i.d. from a categorical peaked on
{all-0, all-1, two 2x2 checkers} with a small uniform noise floor. Tests H1 —
block-factorized reverse models should absorb the local correlations while
pixel-factorized ones cannot.

Single run:

```bash
python train_synthetic.py --block_size 4 --epochs 30 --device cuda
```

Sweep over |G| x seeds:

```bash
python run_e1.py --device cuda --epochs 30 --seeds 42 43 44 --block_sizes 1 4
```

Reports final reconstruction loss and block-state TV distance to the ground truth.

## Project Structure

```
fldd/
  data.py      - binarized MNIST loading
  synthetic.py - E1 synthetic block-tiled dataset + TV metric
  forward.py   - learned forward process (element-wise corruption)
  blocks.py    - block reshape / target / index utilities
  unet.py      - U-Net reverse model with block output head
  train.py     - ELBO loss and training loop
  sample.py    - reverse sampling
```

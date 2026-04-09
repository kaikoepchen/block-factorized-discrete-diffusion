# AML 2026 Project — Block-Factorized Discrete Diffusion

Closing the factorization gap in discrete diffusion via locally-coupled reverse models.

## Setup

```bash
pip install -r requirements.txt
```

## Training

Train the FLDD baseline (pixel-factorized) on binarized MNIST:

```bash
python train_mnist.py --T 4 --epochs 100 --batch_size 128 --lr 3e-4 --seed 42
```

Samples are saved to `samples/` every 10 epochs. Checkpoints go to `checkpoints/`.

## FID Evaluation

After training, compute FID:

```bash
python evaluate_fid.py --checkpoint checkpoints/best.pt --T 4 --n_samples 10000
python -m pytorch_fid fid_stats/real fid_stats/generated
```

## Project Structure

```
fldd/
  data.py      - binarized MNIST loading
  forward.py   - learned forward process (element-wise corruption)
  unet.py      - U-Net reverse model
  train.py     - ELBO loss and training loop
  sample.py    - reverse sampling
```

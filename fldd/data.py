import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


def get_binarized_mnist(data_dir="./data", batch_size=128, num_workers=0):
    """Load MNIST and binarize by thresholding at 0.5.

    Returns train and test dataloaders with images as binary tensors
    of shape (B, 1, 28, 28) with values in {0, 1}.
    """
    transform = transforms.ToTensor()

    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # binarize: threshold at 0.5
    train_imgs = (train_set.data.float() / 255.0 > 0.5).float()
    test_imgs = (test_set.data.float() / 255.0 > 0.5).float()

    # add channel dim: (N, 28, 28) -> (N, 1, 28, 28)
    train_imgs = train_imgs.unsqueeze(1)
    test_imgs = test_imgs.unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(train_imgs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_imgs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader

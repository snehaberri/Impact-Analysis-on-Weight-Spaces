"""Image family: morph between two MNIST or FashionMNIST images.

Uses torchvision's MNIST (28x28 grayscale digits, downloads on first run).
Requires network access to the MNIST mirror — if that's unavailable in your
environment, fall back to the sklearn `load_digits` (8x8) version instead.
"""
import torch
from torchvision import datasets, transforms

_MNIST = datasets.MNIST(root="./data", train=True, download=True,
                         transform=transforms.ToTensor())
_FASHION_MNIST = datasets.FashionMNIST(root="./data", train=True, download=True,
                         transform=transforms.ToTensor())
_TARGETS = _MNIST.targets  # tensor of labels, shape (60000,)
_FASHION_TARGETS = _FASHION_MNIST.targets


def get_digit_image(digit_class: int, idx: int = 0, dataset: str = "mnist") -> torch.Tensor:
    """Return one class example from MNIST or FashionMNIST, shape (28, 28)."""
    if dataset == "mnist":
        images, targets = _MNIST, _TARGETS
    elif dataset in ("fashion", "fashion-mnist", "fashion_mnist"):
        images, targets = _FASHION_MNIST, _FASHION_TARGETS
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion-mnist'")
    idxs = (targets == digit_class).nonzero().flatten()
    img, _ = images[idxs[idx].item()]  # (1, 28, 28), scaled to [0, 1] by ToTensor
    return img.squeeze(0)


def morph(I0: torch.Tensor, I1: torch.Tensor, t: float) -> torch.Tensor:
    return (1 - t) * I0 + t * I1


def image_res(img: torch.Tensor) -> int:
    return img.shape[-1]

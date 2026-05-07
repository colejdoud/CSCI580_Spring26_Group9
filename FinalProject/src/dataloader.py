from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DigitRecord:
    """
    Stores information parsed from a project digit filename.

    Example filename:
        3-4-1.png

    Means:
        label = 3
        group_id = 4
        member_id = 1
    """
    path: Path
    label: int
    group_id: int
    member_id: int


def get_default_transform() -> transforms.Compose:
    """
    The project handout asks us to use:

        transforms.ToTensor()
        transforms.Normalize(mean=0.5, std=0.5)

    ToTensor:
        [0, 255] -> [0, 1]

    Normalize with mean=0.5, std=0.5:
        [0, 1] -> [-1, 1]
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def parse_digit_filename(path: Union[str, Path]) -> DigitRecord:
    """
    Parse filenames with this format:

        <label>-<groupID>-<memberID>.png

    Example:
        7-4-2.png

    Returns:
        DigitRecord(path=..., label=7, group_id=4, member_id=2)
    """
    path = Path(path)

    if path.suffix.lower() != ".png":
        raise ValueError(f"Expected a .png file, got: {path.name}")

    parts = path.stem.split("-")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid filename: {path.name}. "
            "Expected format: <label>-<groupID>-<memberID>.png"
        )

    label_text, group_text, member_text = parts

    if not label_text.isdigit():
        raise ValueError(f"Invalid digit label in filename: {path.name}")

    if not group_text.isdigit():
        raise ValueError(f"Invalid group ID in filename: {path.name}")

    if not member_text.isdigit():
        raise ValueError(f"Invalid member ID in filename: {path.name}")

    label = int(label_text)
    group_id = int(group_text)
    member_id = int(member_text)

    if label < 0 or label > 9:
        raise ValueError(
            f"Invalid label in {path.name}. Label must be between 0 and 9."
        )

    return DigitRecord(
        path=path,
        label=label,
        group_id=group_id,
        member_id=member_id
    )


def load_png_as_grayscale_array(
    path: Union[str, Path],
    strict_size: bool = True
) -> np.ndarray:
    """
    Loads one PNG image as a grayscale NumPy array.

    Expected output:
        shape: (28, 28)
        dtype: uint8
        value range: 0 to 255

    If strict_size=True, the function raises an error when an image
    is not exactly 28x28.
    """
    path = Path(path)

    with Image.open(path) as image:
        image = image.convert("L")

        if image.size != (28, 28):
            if strict_size:
                raise ValueError(
                    f"{path.name} has size {image.size}, but expected (28, 28)."
                )

            image = image.resize((28, 28))

        array = np.asarray(image, dtype=np.uint8)

    return array


def ProjectDataLoader(
    digits_dir: Union[str, Path] = "digits",
    strict_size: bool = True,
    return_metadata: bool = False
):
    """
    Reads all project digit PNG files into NumPy arrays.

    This is the helper function requested in the project handout.

    Basic use:
        images, labels = ProjectDataLoader()

    Returns:
        images:
            NumPy array with shape (N, 28, 28)
            dtype uint8
            values from 0 to 255

        labels:
            NumPy array with shape (N,)
            dtype int64
            values from 0 to 9

    Optional:
        images, labels, metadata = ProjectDataLoader(return_metadata=True)

    metadata is useful later when you need per-group or per-member analysis.
    """
    digits_dir = Path(digits_dir)

    if not digits_dir.exists():
        raise FileNotFoundError(f"Could not find digits directory: {digits_dir}")

    png_files = sorted(digits_dir.glob("*.png"))

    if len(png_files) == 0:
        raise FileNotFoundError(f"No .png files found in: {digits_dir}")

    records = [parse_digit_filename(path) for path in png_files]

    records = sorted(
        records,
        key=lambda record: (
            record.group_id,
            record.member_id,
            record.label,
            record.path.name
        )
    )

    images = []
    labels = []
    metadata = []

    for record in records:
        image_array = load_png_as_grayscale_array(
            record.path,
            strict_size=strict_size
        )

        images.append(image_array)
        labels.append(record.label)

        metadata.append({
            "filename": record.path.name,
            "path": str(record.path),
            "label": record.label,
            "group_id": record.group_id,
            "member_id": record.member_id
        })

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int64)

    if return_metadata:
        return images, labels, metadata

    return images, labels


class ProjectDigitsDataset(Dataset):
    """
    PyTorch Dataset wrapper for the project handwritten digit images.

    This lets you use the project images with a PyTorch DataLoader.
    """

    def __init__(
        self,
        digits_dir: Union[str, Path] = "digits",
        transform: Optional[Callable] = None,
        strict_size: bool = True
    ):
        self.images, self.labels, self.metadata = ProjectDataLoader(
            digits_dir=digits_dir,
            strict_size=strict_size,
            return_metadata=True
        )

        if transform is None:
            transform = get_default_transform()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image_array = self.images[index]
        label = int(self.labels[index])

        image = Image.fromarray(image_array)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def make_project_digit_loader(
    digits_dir: Union[str, Path] = "digits",
    batch_size: int = 64,
    shuffle: bool = False,
    strict_size: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for your project digit images.

    Use this later when testing your trained MLP on the collected images.
    """
    dataset = ProjectDigitsDataset(
        digits_dir=digits_dir,
        transform=get_default_transform(),
        strict_size=strict_size
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def get_mnist_loaders(
    data_dir: Union[str, Path] = "data/mnist",
    batch_size: int = 64,
    validation_fraction: float = 0.1,
    seed: int = 580,
    num_workers: int = 0
):
    """
    Creates PyTorch DataLoaders for MNIST.

    Returns:
        train_loader
        validation_loader
        test_loader

    MNIST is used for training and normal testing.
    Your collected project digits are used for extra project testing.
    """
    data_dir = Path(data_dir)
    transform = get_default_transform()

    full_train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    validation_size = int(len(full_train_dataset) * validation_fraction)
    train_size = len(full_train_dataset) - validation_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, validation_loader, test_loader
import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from dataloader import (
    ProjectDataLoader,
    make_project_digit_loader,
    get_mnist_loaders
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--digits-dir",
        default=str(PROJECT_ROOT / "digits"),
        help="Path to the folder containing project digit PNG files."
    )
    parser.add_argument(
        "--check-mnist",
        action="store_true",
        help="Also test loading MNIST. This may download MNIST the first time."
    )

    args = parser.parse_args()

    print("Checking project digit images...")
    print(f"Digits directory: {args.digits_dir}")

    images, labels, metadata = ProjectDataLoader(
        digits_dir=args.digits_dir,
        return_metadata=True
    )

    print()
    print("Raw NumPy data loaded successfully.")
    print(f"images shape: {images.shape}")
    print(f"images dtype:  {images.dtype}")
    print(f"labels shape: {labels.shape}")
    print(f"labels dtype:  {labels.dtype}")
    print(f"pixel min:     {images.min()}")
    print(f"pixel max:     {images.max()}")

    print()
    print("Label counts:")
    counts = Counter(labels.tolist())

    for digit in range(10):
        print(f"Digit {digit}: {counts.get(digit, 0)} image(s)")

    print()
    print("First few metadata records:")

    for row in metadata[:5]:
        print(row)

    print()
    print("Checking PyTorch DataLoader for project digits...")

    project_loader = make_project_digit_loader(
        digits_dir=args.digits_dir,
        batch_size=8,
        shuffle=True
    )

    batch_images, batch_labels = next(iter(project_loader))

    print("PyTorch batch loaded successfully.")
    print(f"batch_images shape: {batch_images.shape}")
    print(f"batch_images dtype:  {batch_images.dtype}")
    print(f"batch_labels shape: {batch_labels.shape}")
    print(f"batch_labels dtype:  {batch_labels.dtype}")
    print(f"batch pixel min:     {batch_images.min().item():.4f}")
    print(f"batch pixel max:     {batch_images.max().item():.4f}")

    if args.check_mnist:
        print()
        print("Checking MNIST DataLoaders...")

        train_loader, validation_loader, test_loader = get_mnist_loaders(
            data_dir=PROJECT_ROOT / "data" / "mnist",
            batch_size=64
        )

        mnist_images, mnist_labels = next(iter(train_loader))

        print("MNIST batch loaded successfully.")
        print(f"train batches:        {len(train_loader)}")
        print(f"validation batches:   {len(validation_loader)}")
        print(f"test batches:         {len(test_loader)}")
        print(f"mnist_images shape:   {mnist_images.shape}")
        print(f"mnist_labels shape:   {mnist_labels.shape}")


if __name__ == "__main__":
    main()
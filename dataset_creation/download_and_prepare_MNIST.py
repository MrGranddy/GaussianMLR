import os
import argparse
from torchvision.datasets import MNIST
from PIL import Image

def save_images_by_label(dataset, output_dir):
    """Saves images into folders grouped by label."""
    os.makedirs(output_dir, exist_ok=True)

    # Create directories for each label
    for label in range(10):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

    # Save images in the respective label directories
    for index, (image, label) in enumerate(dataset):
        label_dir = os.path.join(output_dir, str(label))
        image_path = os.path.join(label_dir, f"{index}.png")
        image.save(image_path)  # 'image' is a PIL image, which has the 'save' method

def prepare_mnist_dataset(download_dir, prepared_dir):
    """Download and prepare MNIST dataset using torchvision and organize by label."""
    # Download MNIST dataset without transforming to tensor
    train_dataset = MNIST(download_dir, train=True, download=True)
    test_dataset = MNIST(download_dir, train=False, download=True)

    # Save images grouped by label
    train_output_dir = os.path.join(prepared_dir, 'train')
    test_output_dir = os.path.join(prepared_dir, 'test')

    print("Organizing training data...")
    save_images_by_label(train_dataset, train_output_dir)
    
    print("Organizing testing data...")
    save_images_by_label(test_dataset, test_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare MNIST dataset")
    parser.add_argument("download_dir", type=str, help="Directory to download raw MNIST data files")
    parser.add_argument("prepared_dir", type=str, help="Directory to save the prepared MNIST dataset")
    args = parser.parse_args()

    prepare_mnist_dataset(args.download_dir, args.prepared_dir)

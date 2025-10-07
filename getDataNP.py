import gzip
import struct
from typing import Tuple

import numpy as np


def display_images(images: np.ndarray) -> None:
    """Pretty-print images (uint8) similar to Archive/getData.py.
    Expects a NumPy array shaped (N, rows, cols) with dtype uint8.
    """
    # Iterate like the original for consistent output formatting
    for array in images:
        for row in array:
            # row is a 1D numpy array; format each pixel as width-3 integer
            print("".join(f"{int(item):3}" for item in row))



def get_data(count: int, labels_path: str, images_path: str,):
    """
    Returns (labels, images):
    - labels: shape (count,), dtype uint8
    - images: shape (count, rows, cols), dtype uint8
    The function clamps `count` to the number of available items in the files.
    """
    if count <= 0:
        return np.empty((0,), dtype=np.uint8), np.empty((0, 28, 28), dtype=np.uint8)

    if count > 60000:
        count = 60000  # MNIST training set size
        print("Using max size of 60000")

    # Read labels
    with gzip.open(labels_path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("Labels file header is incomplete or missing.")
        magic, num_items = struct.unpack(">II", header)
        # IDX magic for labels is typically 2049, but we won't enforce strictly.
        labels_buf = f.read(count)
        if len(labels_buf) != count:
            raise ValueError("Labels file does not contain enough data.")
        labels = np.frombuffer(labels_buf, dtype=np.uint8)

    # Read images
    with gzip.open(images_path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("Images file header is incomplete or missing.")
        magic, num_images, rows, cols = struct.unpack(">IIII", header)
        # IDX magic for images is typically 2051.
        img_bytes = rows * cols * count
        images_buf = f.read(img_bytes)
        if len(images_buf) != img_bytes:
            raise ValueError("Images file does not contain enough data.")
        images = np.frombuffer(images_buf, dtype=np.uint8).reshape(count, rows, cols)

    return labels, images


if __name__ == "__main__":
    labels_path = "files/train-labels-idx1-ubyte.gz"
    images_path = "files/train-images-idx3-ubyte.gz"
    lbls, imgs = get_data(40, labels_path, images_path)
    print(f"Loaded {len(lbls)} labels and {len(imgs)} images")
    print(f"Image shape: {imgs.shape}")
    display_images(imgs)


"""Helper script to downsample images.

Usage example:
    python downsample.py my_image.png 300
This will create an image my_image_down.png with 300 pixels in the larger dimension.

Author: Leonard Bruns (2019)
"""

import argparse
import os

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt


def main() -> None:
    """Downsample image specified by command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("out_pixels")
    args = parser.parse_args()

    image = Image.open(args.image)
    image = np.array(image)

    rows = image.shape[0]
    cols = image.shape[1]

    logic_indices = np.zeros_like(image, dtype=np.bool)

    if rows > cols:
        row_indices = np.around(np.linspace(0, rows - 1, num=args.out_pixels)).astype(
            np.int
        )
        col_indices = row_indices[row_indices < cols]
    else:
        col_indices = np.around(np.linspace(0, cols - 1, num=args.out_pixels)).astype(
            np.int
        )

        row_indices = col_indices[col_indices < rows]

    logic_rows = np.zeros_like(image, dtype=np.bool)
    logic_rows[row_indices, :, :] = True
    logic_cols = np.zeros_like(image, dtype=np.bool)
    logic_cols[:, col_indices, :] = True

    logic_indices = np.logical_and(logic_rows, logic_cols)

    downsampled_image = image[logic_indices].reshape(
        len(row_indices), len(col_indices), 3
    )
    plt.imshow(downsampled_image)
    plt.show()

    out_im = Image.fromarray(downsampled_image)

    path_parts = os.path.splitext(args.image)
    out_im.save(path_parts[0] + "_down" + path_parts[1])


if __name__ == "__main__":
    main()

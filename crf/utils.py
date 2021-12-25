"""Utility functions for image and segmentation handling.

Author: Leonard Bruns (2019)
"""
import typing

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path: str) -> np.ndarray:
    """Load an image at a given path as a numpy array.

    Args:
        image_path: Path of the image.

    Returns:
        The loaded image array. Shape (H,W,C).
    """
    image = Image.open(image_path)
    return np.array(image)


def load_segmentation(
    segmentation_path: str, unknown: typing.Optional[np.array] = None
) -> np.ndarray:
    """Load a segmentation at a given path as a numpy array.

    Args:
        segmentation_path: Path of the segmentation.
        unknown:
            Color of unlabeled, will be assigned index 0, if none, index 0 still
            correponds to unknown.

    Returns:
        Array containing the segmentation. Shape (H,W).
    """
    seg_rgb = np.array(Image.open(segmentation_path))
    rows = seg_rgb.shape[0]  # pylint: disable=E1136  # pylint/issues/3139
    cols = seg_rgb.shape[1]  # pylint: disable=E1136  # pylint/issues/3139
    seg_colors = seg_rgb.reshape(
        -1, seg_rgb.shape[2]
    )  # pylint: disable=E1136  # pylint/issues/3139

    colors = np.unique(seg_colors, axis=0)
    seg = np.empty(
        seg_colors.shape[0], dtype=np.uint8
    )  # pylint: disable=E1136  # pylint/issues/3139

    # swap unknown color s.t. it is at index 0
    if unknown is not None:
        if unknown.tolist() in colors.tolist():
            index = np.argmax(np.all(unknown == colors, 1))
            colors[[0, index]] = colors[[index, 0]]
        else:
            colors = np.vstack((unknown.reshape((1, -1)), colors))
            print("no unknown")

    for color_id, color in enumerate(colors):
        if unknown is None:
            seg[np.all(seg_colors == color, 1)] = color_id + 1
        else:
            seg[np.all(seg_colors == color, 1)] = color_id

    return seg.reshape(rows, cols)


def plot_image(image: np.ndarray) -> None:
    """Show an image without axis.

    Args:
        image: The image to show. Shape (H,W,C).
    """
    figure = plt.figure()
    plt.axis("off")
    plt.imshow(image)
    figure.show()


def plot_segmentation(
    segmentation: np.ndarray,
    image: typing.Optional[np.ndarray] = None,
    alpha: typing.Optional[float] = 0.5,
    classes: typing.Optional[int] = None,
    saveas: typing.Optional[str] = None,
) -> None:
    """Plot a segmentation.

    Args:
        segmentation:
            Either 2D array of integers representing the class or 3D array, with last
            dimension being the probabilities of each class.
            Shape (H,W) or (H,W,N), respectively.
        image: Show this image with opacity of alpha in front of segmentation.
        alpha: Opacity for overlayed image.
        classes: Number of classes (not counting unknown).
        saveas: The path to save the image to.
    """
    figure = plt.figure()
    plt.axis("off")

    # plot segmentation
    if segmentation.ndim == 2:
        plt.imshow(segmentation, vmin=0, vmax=classes)
    elif segmentation.ndim == 3:
        plt.imshow(np.argmax(segmentation, axis=2) + 1, vmin=0, vmax=classes)

    # plot image
    if image is not None:
        plt.imshow(image, alpha=alpha)

    if saveas is not None:
        if image is not None:
            plt.savefig(saveas, bbox_inches="tight", pad_inches=0.0)
        else:
            if segmentation.ndim == 2:
                plt.imsave(saveas, segmentation, vmin=0, vmax=classes)
            elif segmentation.ndim == 3:
                plt.imsave(
                    saveas, np.argmax(segmentation, axis=2) + 1, vmin=0, vmax=classes
                )

    figure.show()

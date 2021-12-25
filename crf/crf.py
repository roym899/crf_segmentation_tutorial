"""Module for inference in fully-connected CRFs.

Author: Leonard Bruns (2019)
"""

import typing
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class CrfParameters:
    """Parameters for fully connected CRF with Gaussian kernels."""

    # Weights for each kernel.
    kernel_weights: typing.List = field(default_factory=lambda: [10, 5])
    # Spatial standard deviation for appearance kernel.
    theta_alpha: float = 60
    # Color standard deviation for appearance kernel.
    theta_beta: float = 10
    # Spatial standard deviation for smoothness kernel.
    theta_gamma: float = 1
    # Whether to use naive or efficient message passing.
    efficient_message_passing: bool = True
    # Factor to downsample the spatial dimensions for the 5D representation.
    spatial_downsampling: float = 15
    # Factor to downsample the range dimensions for the 5D representation.
    range_downsampling: float = 15
    # Number of iterations to perform.
    iterations: int = 3


def appearance_kernel(
    x_1: int,
    y_1: int,
    p_1: np.ndarray,
    x_2: int,
    y_2: int,
    p_2: np.ndarray,
    theta_alpha: float,
    theta_beta: float,
) -> float:
    """Compute appearance kernel.

    Args:
        x_1: X coordinate of first pixel.
        y_1: Y coordinate of first pixel.
        p_1: Color vector of first pixel.
        x_2: X coordinate of second pixel.
        y_2: Y coordinate of second pixel.
        p_2: Color vector of second pixel.
        theta_alpha: Standard deviation for the position.
        theta_beta: Standard deviation for the color.

    Returns:
        The output of the appearence kernel.
    """
    return np.exp(
        -((x_1 - x_2) ** 2.0 + (y_1 - y_2) ** 2.0) / (2 * theta_alpha ** 2.0)
        - np.sum((p_1 - p_2) ** 2.0) / (2.0 * theta_beta ** 2.0)
    )


def smoothness_kernel(
    x_1: int,
    y_1: int,
    p_1: np.ndarray,
    x_2: int,
    y_2: int,
    p_2: np.ndarray,
    theta_gamma: float,
) -> float:
    """Compute smoothness kernel.

    Args:
        x_1: X coordinate of first pixel.
        y_1: Y coordinate of first pixel.
        p_1: Color vector of first pixel.
        x_2: X coordinate of second pixel.
        y_2: Y coordinate of second pixel.
        p_2: Color vector of second pixel.
        theta_gamma: Standard deviation for the position.

    Returns:
        The output of the smoothness kernel.
    """
    del p_1, p_2
    return np.exp(
        -((x_1 - x_2) ** 2.0 + (y_1 - y_2) ** 2.0) / (2.0 * theta_gamma ** 2.0)
    )


def normalize(potentials: np.ndarray) -> np.ndarray:
    """Normalize potentials such that output is a valid pixelwise distribution.

    Args:
        potentials: Array of potentials. Shape (H,W,N).

    Returns:
        Probability array with same shape as potentials.
        Probabilities sum up to 1 at every slice (i,j,:).
    """
    # TODO: implement normalization here
    pass


def message_passing(
    image: np.ndarray,
    current_probabilities: np.ndarray,
    kernel_functions: typing.List[
        typing.Callable[[int, int, np.ndarray, int, int, np.ndarray], float]
    ],
) -> np.ndarray:
    """Perform "message passing" as the first step of the inference loop.

    Args:
        image:
            Array of size ROWS x COLUMNS x CHANNELS, representing the image used to
            compute the kernel.
        current_probabilities:
            Array of size ROWS x COLUMNS x CLASSES, representing the current
            probabilities.
        kernel_functions: The kernel functions defining the edge potential.

    Returns:
        Array of size ROWS x COLUMNS x CLASSES x KERNELS, representing the intermediate
        result of message passing for each kernel.
    """
    # naive version
    rows = image.shape[0]
    cols = image.shape[1]
    classes = current_probabilities.shape[2]
    result = np.zeros(
        (
            current_probabilities.shape[0],
            current_probabilities.shape[1],
            current_probabilities.shape[2],
            len(kernel_functions),
        ),
        dtype=float,
    )

    # TODO implement naive message passing (using loops)


def efficient_message_passing(
    image: np.ndarray,
    current_probabilities: np.ndarray,
    spatial_downsampling: float,
    range_downsampling: float,
    theta_alpha: float,
    theta_beta: float,
    theta_gamma: float,
) -> np.ndarray:
    """Perform efficient "message passing" by downsampling and convolution in 5D.

    This assumes two kernels: an appearance kernel based on theta_alpha and theta_beta,
    and a smoothness kernel based on theta_gamma.

    Args:
        image:
            Array of size ROWS x COLUMNS x CHANNELS, representing the image used to
            compute the kernel.
        current_probabilities:
            Array of size ROWS x COLUMNS x CLASSES, representing the current
            probabilities.
        spatial_downsampling:
            Factor to downsample the spatial dimensions for the 5D representation.
        range_downsampling:
            Factor to downsample the range dimensions for the 5D representation.
        theta_alpha: Spatial standard deviation for the appearance kernel.
        theta_beta: Color standard deviation for the appearance kernel.
        theta_gamma: Spatial stadnard deviation for the smoothness kernel.

    Returns:
        Array of size ROWS x COLUMNS x CLASSES x KERNELS, representing the intermediate
        result of message passing for each kernel.
    """
    t_0 = time.time()

    rows = image.shape[0]
    cols = image.shape[1]
    classes = current_probabilities.shape[2]
    color_range = 255

    ds_rows = int(np.ceil(rows / spatial_downsampling))
    ds_cols = int(np.ceil(cols / spatial_downsampling))
    ds_range = int(np.ceil(color_range / range_downsampling))

    print(f"Downsampled to: {ds_rows}x{ds_cols}x{ds_range}")

    result = np.zeros(
        (
            current_probabilities.shape[0],
            current_probabilities.shape[1],
            current_probabilities.shape[2],
            2,
        ),
        dtype=float,
    )

    # precompute indices
    indices_list = []
    for row in np.arange(rows):
        for col in np.arange(cols):
            indices_list.append(
                (row, col, image[row, col, 0], image[row, col, 1], image[row, col, 2])
            )
    indices_list = np.array(indices_list, dtype=np.float)
    indices_list[:, 0:2] = indices_list[:, 0:2] / float(spatial_downsampling)
    indices_list[:, 2:] = indices_list[:, 2:] / float(range_downsampling)
    indices_list = np.round(indices_list).astype(np.int)

    for class_id in np.arange(classes):
        # allocate 5D feature space
        feature_space = np.zeros((ds_rows+1, ds_cols+1, ds_range+1, ds_range+1, ds_range+1))

        # downsample with box filter and go to 5D space at same time
        # Hint: use indices list for this and only loop over row and col

        # TODO: implement downsampling here

        for kernel_id in np.arange(2):
            if kernel_id == 0: # appearance kernel
                # TODO: apply appearance kernel as a gaussian filter
                # Hint 1: use gaussian_filter from scipy.ndimage
                # Hint 2: remember to adjust parameters for downsampling

            if kernel_id == 1: # smoothness kernel
                # TODO: apply smoothness kernel as a gaussian filter
                # Hint 1: use gaussian_filter from scipy.ndimage
                # Hint 2: do we need a 5D convolution for the smoothness kernel?

            # upsample with simple look up (no interpolation for simplicity)
            # TODO: implement upsample here, reverse to downsample
            # Hint: do we actually need this for both kernels? move it if not

    t_1 = time.time()
    print(f"Efficient message passing took {t_1-t_0}s")

    return result


def compatibility_transform(
    q_tilde: np.ndarray, weights: typing.List[float]
) -> np.ndarray:
    """Perform compatability transform as part of the inference loop.

    Args:
        q_tilde:
            Array of size ROWS x COLUMNS x CLASSES x KERNELS, representing the
            intermediate result of message passing for each kernel.
        weights: Weights of each kernel.

    Returns:
        Array of size ROWS x COLUMNS x CLASSES, representing the result after combining
        the kernels and applying the label compatability function (here: Potts model).
    """

    # TODO: implement compatability transform (try with matrix operations only)
    pass

def local_update(q_hat: np.ndarray, unary_potential: np.ndarray):
    """Perform local update as part of the interefence loop.

    Args:
        q_hat:
            Array of size ROWS x COLUMNS x CLASSES, representing the intermediate result
            after combining the kernels and applying the label compatability function.
        unary_potential:
            Array of size ROWS x COLUMNS x CLASSES, representing the prior energy for
            each pixel and class from a different source.
    Returns:
        Array of size ROWS x COLUMNS x CLASSES, representing the probabilities for each
        pixel and class.
    """
    # TODO: implement local update
    pass


def inference(
    image: np.ndarray, initial_probabilities: np.ndarray, params: CrfParameters
) -> np.ndarray:
    """Perform inference in fully connected CRF with Gaussian edge potentials.

    Args:
        image:
            Array of size ROWS x COLUMNS x CHANNELS, representing the image used the
            features.
        initial_probabilities:
            Initial pixelwise probabilities for each class. Used to initialize unary
            potential.
        params:
            Parameter class for fully connected CRFs (see CrfParameters documentation).
    Return:
        Array of size ROWS x COLS x CLASSES
    """
    # define kernels
    kernels = [
        lambda x1, y1, p1, x2, y2, p2: appearance_kernel(
            x1, y1, p1, x2, y2, p2, params.theta_alpha, params.theta_beta
        ),
        lambda x1, y1, p1, x2, y2, p2: smoothness_kernel(
            x1, y1, p1, x2, y2, p2, params.theta_gamma
        ),
    ]

    # initialize
    current_probabilities = initial_probabilities

    unary_potential = -np.log(current_probabilities)

    for _ in np.arange(params.iterations):
        if params.efficient_message_passing:
            q_tilde = efficient_message_passing(
                image,
                current_probabilities,
                spatial_downsampling=params.spatial_downsampling,
                range_downsampling=params.range_downsampling,
                theta_alpha=params.theta_alpha,
                theta_beta=params.theta_beta,
                theta_gamma=params.theta_gamma,
            )
        else:
            q_tilde = message_passing(image, current_probabilities, kernels)
        q_hat = compatibility_transform(q_tilde, params.kernel_weights)
        unnormalized_probabilities = local_update(q_hat, unary_potential)
        current_probabilities = normalize(unnormalized_probabilities)

    return current_probabilities


def test() -> None:
    """Runs simple tests to check functions in this file."""
    test_input = np.array([[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.4], [0.2, 0.3]]])
    test_expected = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.2, 0.8], [0.4, 0.6]]])
    test_out = normalize(test_input)

    assert np.all(test_expected == test_out)
    print("Test of normalize successful.")


if __name__ == "__main__":
    test()

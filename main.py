#!/usr/bin/python3
"""Entry point for tutorial.

Author: Leonard Bruns (2019)
"""
import argparse
import typing

import numpy as np

import crf


def segmentation_to_probabilities(
    segmentation: np.ndarray, prior_probability: float = 0.7
) -> np.ndarray:
    """Convert a segmentation into unary probabilities.

    Converts a pixelwise segmentation into probabilities for each class. The
    prior_probability defines the probabiltiy of the specified class. The remaining
    probability mass is uniformly distributed across the remaining classes.

    Args:
        segmentation:
            2D integer array, representing the class per pixel. 0 indicates unknown.
        prior_probability: Probability assigned to the specified class.
    """
    classes = np.amax(segmentation)
    rows = segmentation.shape[0]
    cols = segmentation.shape[1]
    segm = segmentation.reshape(-1)
    probabilities = np.zeros((segm.shape[0], classes), dtype=np.float)

    for class_id in np.arange(start=0, stop=classes + 1):
        # unknown -> uniform probabilities
        if class_id == 0:
            probabilities[segm == class_id, :] = 1.0 / classes
        else:
            probabilities[segm == class_id, class_id - 1] = prior_probability
            probabilities[segm == class_id, 0 : class_id - 1] = (
                1 - prior_probability
            ) / (classes - 1)
            probabilities[segm == class_id, class_id:] = (1 - prior_probability) / (
                classes - 1
            )
    return probabilities.reshape((rows, cols, classes))


def refine_mode(
    image_path: str, segmentation_path: str, output_path: typing.Optional[str] = None
) -> None:
    """Run the refine mode.

    Takes an image and an initilal segmentation to produce a refined segmentation.

    Args:
        image_path: Path of the image.
        segmentation_path: Path of the initial segmentation.
        output_path: Path to save the refined segmentation to.
    """
    image = crf.load_image(image_path)
    initial_segmentation = crf.load_segmentation(
        segmentation_path, unknown=np.array([0, 0, 0])
    )

    crf.plot_segmentation(initial_segmentation)

    crf.plot_image(image)

    initial_probabilities = segmentation_to_probabilities(initial_segmentation, 0.7)

    params = crf.CrfParameters()

    # low res message passing
    params.kernel_weights = [
        10,
        1,
    ]  # smoothness kernel not helpful for such a coarse image
    params.efficient_message_passing = False

    # efficient message passing
    # params.kernel_weights = [10,5] # smoothness kernel more useful now
    # params.efficient_message_passing = True

    final_probabilities = crf.inference(image, initial_probabilities, params)

    crf.plot_segmentation(final_probabilities, image=image, alpha=0.5)
    crf.plot_segmentation(final_probabilities, saveas=output_path)

    print("Press return to quit...")
    input()


def main() -> None:
    """Execute CRF-based segmentation refinement from commandline arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    parser_refine = subparsers.add_parser("refine")

    parser_refine.add_argument("image")
    parser_refine.add_argument("segmentation")
    parser_refine.add_argument(
        "--output",
        "-o",
        help="Output path for the refined segmentation. "
        "If none is passed, the output will only be shown.",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    if args.mode == "refine":
        refine_mode(args.image, args.segmentation, args.output)


if __name__ == "__main__":
    main()

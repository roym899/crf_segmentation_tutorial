# Fast and Accurate Image Segmentation using Fully Connected Conditional Random Fields

This tutorial was created for a course on probabilistic graphical models at KTH. Start by reading the tutorial in [tutorial.pdf](tutorial.pdf). Only the inference functions as stated in the tutorial have to be implemented. A number of Python packages are required for this tutorial. Make sure all of them are installed and available in your Python environment (you will need Python >3.7, since I use `dataclass`).

Versions/Packages used for development and testing:
```
Python          3.7.5
numpy           1.17.3
matplotlib      3.1.1
Pillow          6.2.0     
scipy           1.3.1        
```

All functions have documentation (type hints + docstring). Make sure to read and understand these before implementing a function. A few images from the [MSRC-9 dataset](http://research.microsoft.com/vision/cambridge/recognition/) dataset have been included. Download this dataset if you want to try out more images. Most other formats, where the segmentation is available as an image should also work out of the box ([COCO](http://cocodataset.org/#home) has been tested, but is less interesting, since the labels are generally already much more accurate).

## Overview
### `main.py`
Entry point to the program. This uses the crf package, which we develop in this tutorial. Currently there is one way of running the program.

#### Refine Mode
This takes as input an image and a rough segmentation. The improved segmentation will be either shown or written to disk.
```console
python3 main.py refine image segmentation -o output
```

### `crf/crf.py`
This is the file that contains the functions you will implement. It contains code skeletons and comments guiding you through the implementation. All interfaces to the functions are defined and main.py ensures that the functions are called in the right way.

### `crf/utils.py`
This file contains helper functions, like loading images and segmentations and bringing them to the right format. Tested with [MSRC-9 dataset](http://research.microsoft.com/vision/cambridge/recognition/).
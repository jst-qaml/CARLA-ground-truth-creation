# CARLA Simulator Data Collection

## Overview

This branch provides a set of utilities for data collection in the CARLA simulator using various sensors. The primary goal is to collect data, such as RGB images, depth maps, and semantic segmentation, from the CARLA environment. These datasets can be used for a variety of tasks, including autonomous driving, machine learning, and computer vision research.

## Requirement

CARLA version: 0.9.14

### Required Packages

To run the scripts in this branch, ensure the following packages are installed. You can install most of the packages via `pip` using the instructions below.

1. **CARLA Simulator**:
   - Package: `carla`
   - **Installation**:
     - Download the appropriate `.egg` file from CARLA and place it in the project directory.
     - Follow the [CARLA documentation](https://carla.readthedocs.io/en/latest/build_linux/) for setup instructions.

2. **OpenCV for Python**:
   - Package: `opencv-python`
   - **Installation**:
     ```bash
     pip install opencv-python
     ```

3. **Numpy**:
   - Package: `numpy`
   - **Installation**:
     ```bash
     pip install numpy
     ```

4. **Matplotlib**:
   - Package: `matplotlib`
   - **Installation**:
     ```bash
     pip install matplotlib
     ```

5. **Pygame**:
   - Package: `pygame`
   - **Installation**:
     ```bash
     pip install pygame
     ```

6. **PyCocoTools**:
   - Package: `pycocotools`
   - **Installation**:
     ```bash
     pip install pycocotools
     ```

7. **Scipy** (for spatial distance calculations):
   - Package: `scipy`
   - **Installation**:
     ```bash
     pip install scipy
     ```

8. **Pillow** (for image grabbing and processing):
   - Package: `pillow`
   - **Installation**:
     ```bash
     pip install pillow
     ```

9. **Detectron2**:
   - Package: `detectron2`
   - **Installation**:
     Detectron2 has specific installation instructions depending on your system and hardware. Please refer to the official [Detectron2 Installation Guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

10. **Scikit-Image**:
    - Package: `scikit-image`
    - **Installation**:
      ```bash
      pip install scikit-image
      ```

11. **Queue**:
    - **Note**: This is part of the Python standard library. No installation is required.

12. **Contextlib**:
    - **Note**: This is part of the Python standard library. No installation is required.

### Installation

To install the required packages, use the following command:

```bash
pip install opencv-python numpy matplotlib pygame pycocotools scipy pillow scikit-image
```

## Usage

The main script, `manual_data_collect.py`, allows for manual control of a bicycle in the CARLA simulator and collects data using different sensors. The collected data includes RGB images, depth maps, and semantic segmentation, which are saved in separate directories for easy access.

### Manual Data Collection

`manual_data_collect.py` is used to manually control the bicycle and collect sensor data. Below are the instructions for using the script:

### Controls

- **Start Image Collection**: Press `P` to start saving the images.
- **Bicycle Control**: Use the following keys for movement:
  - `W` - Move forward
  - `S` - Move backward
  - `A` - Turn left
  - `D` - Turn right
- **Exit the Program**: Press `U` to exit the program.

### Output

Once the image collection is started, the script will save the collected data into newly created directories under the current working path. The directories are named using the current timestamp in the following format:
- `{timestamp}_rgb` - for RGB images
- `{timestamp}_depth` - for depth maps
- `{timestamp}_seg` - for semantic segmentation images

### Image Resolution

The default image resolution is set to `1920x1080`. However, this can be adjusted by modifying the following variables in the script:

```python
IM_WIDTH = 1920
IM_HEIGHT = 1080
```

## Others

Dep-cam.py, RGB-cam.py and Seg-cam.py are used to collect data separately (RGB & Depth & Semantic).
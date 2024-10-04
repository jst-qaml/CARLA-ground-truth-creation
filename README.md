# CARLA Simulator Data Collection

## Overview

This branch provides a set of utilities for data collection in the CARLA simulator using various sensors. The primary goal is to collect data, such as RGB images, depth maps, and semantic segmentation, from the CARLA environment. These datasets can be used for a variety of tasks, including autonomous driving, machine learning, and computer vision research.

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

import glob
import os
import sys
import random
import numpy as np
import cv2
import carla
import skimage.measure as measure
import queue
from contextlib import contextmanager

# Constants
IM_WIDTH = 640
IM_HEIGHT = 480

# Function to process semantic segmentation images and extract bounding boxes
def process_segmentation_image(image_seg, object_list):
    img_semseg_bgr = cv2.cvtColor(np.array(image_seg.raw_data).reshape((IM_HEIGHT, IM_WIDTH, 4)), cv2.COLOR_RGBA2BGR)
    img_semseg_hsv = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGR2HSV)

    object_bboxes = {}
    for obj_name, rgb_value in object_list.items():
        mask = get_mask(img_semseg_hsv, rgb_value)
        bboxes = get_bbox_from_mask(mask)
        object_bboxes[obj_name] = bboxes

    return object_bboxes

# Function to get mask for a specific object category
def get_mask(seg_im, rgb_value):
    hsv_value = cv2.cvtColor(rgb_value, cv2.COLOR_RGB2HSV)
    hsv_low = np.array([[[hsv_value[0][0][0]-5, hsv_value[0][0][1], hsv_value[0][0][2]-5]]])
    hsv_high = np.array([[[hsv_value[0][0][0]+5, hsv_value[0][0][1], hsv_value[0][0][2]+5]]])
    mask = cv2.inRange(seg_im, hsv_low, hsv_high)
    return mask

# Function to get bounding boxes from a mask
def get_bbox_from_mask(mask):
    label_mask = measure.label(mask)
    props = measure.regionprops(label_mask)
    return [prop.bbox for prop in props]

# Context manager for creating actors
@contextmanager
def create_actor(world, blueprint, transform, attach_to=None):
    actor = world.spawn_actor(blueprint, transform, attach_to=attach_to)
    try:
        yield actor
    finally:
        if actor is not None:
            actor.destroy()

# Main script
try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(200.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Spawning the main vehicle
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())

    with create_actor(world, vehicle_bp, spawn_point) as vehicle:
        # Spawning RGB camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        with create_actor(world, camera_bp, camera_transform, attach_to=vehicle) as camera, \
             create_actor(world, blueprint_library.find('sensor.camera.semantic_segmentation'), camera_transform, attach_to=vehicle) as camera_seg:

            # Queues for image capturing
            image_queue = queue.Queue()
            image_queue_seg = queue.Queue()
            camera.listen(image_queue.put)
            camera_seg.listen(image_queue_seg.put)

            # Spawning 50 random cars
            for _ in range(50):
                vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
                spawn_point = random.choice(world.get_map().get_spawn_points())
                npc = world.try_spawn_actor(vehicle_bp, spawn_point)
                if npc:
                    npc.set_autopilot(True)

            # Object list for semantic segmentation
            object_list = {
                'car': np.uint8([[[0, 0, 142]]])
                # Add more objects as needed
            }

            # Image capturing and processing
            for _ in range(30):  # Capture 30 frames
                image_rgb = image_queue.get()
                image_seg = image_queue_seg.get()

                frame_number = image_rgb.frame
                rgb_image_path = f"test_images/rgb_{frame_number}.png"
                image_rgb.save_to_disk(rgb_image_path)

                # Process segmentation image and get bounding boxes
                bboxes = process_segmentation_image(image_seg, object_list)

                # Crop RGB images based on bboxes
                for obj_name, obj_bboxes in bboxes.items():
                    rgb_image = cv2.imread(rgb_image_path)
                    for bbox in obj_bboxes:
                        minr, minc, maxr, maxc = bbox
                        cropped_image = rgb_image[minr:maxr, minc:maxc]
                        cropped_image_path = f"cropped_images/cropped_{obj_name}_{frame_number}.png"
                        cv2.imwrite(cropped_image_path, cropped_image)

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    actors = [actor for actor in world.get_actors() if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor")]
    for actor in actors:
        actor.destroy()

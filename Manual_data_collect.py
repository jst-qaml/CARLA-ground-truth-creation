import glob
import os
import sys
import random
import numpy as np
import cv2
import datetime
import pygame

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

IM_WIDTH = 1920
IM_HEIGHT = 1080
dt_now = datetime.date.today()
save_images = False
exit_program = False

# Dictionary to store images by frame number
image_store = {}

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3 / 255.0

def save_images_from_frame(frame_number):
    global image_store
    if frame_number in image_store and len(image_store[frame_number]) == 3:
        for sensor_type, image in image_store[frame_number].items():
            if sensor_type == 'rgb':
                folder = f'{dt_now}_rgb'
            elif sensor_type == 'depth':
                folder = f'{dt_now}_depth'
            elif sensor_type == 'semseg':
                folder = f'{dt_now}_seg'
            os.makedirs(folder, exist_ok=True)
            image.save_to_disk(f'{folder}/{sensor_type}_{frame_number:06d}.png')
        del image_store[frame_number]

def store_image(image, sensor_type, color_converter=None):
    global image_store, save_images
    if save_images:
        if color_converter:
            image.convert(color_converter)
        if image.frame not in image_store:
            image_store[image.frame] = {}
        image_store[image.frame][sensor_type] = image
        save_images_from_frame(image.frame)

actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(2000.0)
    world = client.load_world('Town03')
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("crossbike")[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(False)  # Disable autopilot for manual control
    actor_list.append(vehicle)

    # Init RGB cam
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    # Init Depth cam
    camera_depth = blueprint_library.find('sensor.camera.depth')
    camera_depth.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_depth.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_depth.set_attribute("fov", "110")

    # Init SemSeg cam
    camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_semseg.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_semseg.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_semseg.set_attribute("fov", "110")

    # First view
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    sensor_dp = world.spawn_actor(camera_depth, spawn_point, attach_to=vehicle)
    sensor_semseg = world.spawn_actor(camera_semseg, spawn_point, attach_to=vehicle)

    actor_list.append(sensor)
    actor_list.append(sensor_dp)
    actor_list.append(sensor_semseg)

    sensor.listen(lambda image: store_image(image, 'rgb'))
    sensor_dp.listen(lambda image: store_image(image, 'depth', carla.ColorConverter.LogarithmicDepth))
    sensor_semseg.listen(lambda image: store_image(image, 'semseg', carla.ColorConverter.CityScapesPalette))

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                            carla.Rotation(pitch=-90)))

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("CARLA Manual Control")

    # Set up the control object and clock
    control = carla.VehicleControl()
    clock = pygame.time.Clock()
    done = False

    while not exit_program:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    save_images = not save_images
                    print(f"Save images: {save_images}")
                elif event.key == pygame.K_u:
                    exit_program = True

        # Get keyboard input and handle it
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            control.throttle = min(control.throttle + 0.05, 1.0)
        else:
            control.throttle = 0.0

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            control.brake = min(control.brake + 0.2, 1.0)
        else:
            control.brake = 0.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            control.steer = max(control.steer - 0.05, -1.0)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            control.steer = min(control.steer + 0.05, 1.0)
        else:
            control.steer = 0.0

        control.hand_brake = keys[pygame.K_SPACE]

        # Apply the control to the ego vehicle and tick the simulation
        vehicle.apply_control(control)
        world.tick()

        # Update the display
        pygame.display.flip()
        pygame.display.update()

        # Sleep to ensure consistent loop timing
        clock.tick(60)

finally:
    for actor in actor_list:
        actor.destroy()
    pygame.quit()
    print("All cleaned up!")

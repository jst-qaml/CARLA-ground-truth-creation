import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import datetime
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
IM_WIDTH = 640
IM_HEIGHT = 480
dt_now = datetime.date.today()
def process_img(image):
    i = np.array(image.raw_data)
    print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i3/255.0

actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(200.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x = 2.5, z = 0.7))
    spawn_point2 = carla.Transform(carla.Location(x = -5.5, z = 2.5), carla.Rotation(pitch=8.0))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    sensor2 = world.spawn_actor(cam_bp, spawn_point2, attach_to=vehicle)
    actor_list.append(sensor)
    actor_list.append(sensor2)
    sensor.listen(lambda image:image.save_to_disk(f'{dt_now}/%06d.png' % image.frame))
    sensor2.listen(lambda image:image.save_to_disk(f'{dt_now}/%06d.png' % image.frame))

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
    carla.Rotation(pitch=-90)))
    time.sleep(10)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
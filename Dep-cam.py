import carla
import math

# Connect to the simulator
client = carla.Client('localhost', 2000)
world = client.get_world()

# Set up your sensors (camera, etc.)
camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
camera_location = carla.Location(x=1.5, y=0, z=2.4)  # Adjust to your car's specific mounting position
camera_rotation = carla.Rotation(0, 0, 0)
camera_transform = carla.Transform(camera_location, camera_rotation)

# Spawn the camera by attaching it to your vehicle
camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=my_vehicle)

def is_close_to_traffic_sign(vehicle, sign, threshold_in_meters):
    vehicle_location = vehicle.get_location()
    sign_location = sign.get_location()
    distance = math.sqrt(
        (vehicle_location.x - sign_location.x)**2 +
        (vehicle_location.y - sign_location.y)**2 +
        (vehicle_location.z - sign_location.z)**2
    )
    return distance < threshold_in_meters

# Logic to capture an image when close to a traffic sign
def capture_image():
    traffic_signs = world.get_actors().filter('*traffic*sign*')
    for sign in traffic_signs:
        if is_close_to_traffic_sign(my_vehicle, sign, 10):  # 10 meters threshold
            # The function to save the camera image
            camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

# Continuously check for proximity to traffic signs
while True:
    capture_image()

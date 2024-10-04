#Import and Constants
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pycocotools
import pycocotools.mask
import math
import json
import base64
from scipy.spatial import distance
from PIL import ImageGrab
import carla
import skimage.measure as measure
import queue
from detectron2.structures import BoxMode
IM_WIDTH = 640
IM_HEIGHT = 480
dt_now = datetime.date.today()
actor_list = []



object_id = {"None": 0,
             "Buildings": 1,
             "Fences": 2,
             "Other": 3,
             "Pedestrians": 4,
             "Poles": 5,
             "RoadLines": 6,
             "Roads": 7,
             "Sidewalks": 8,
             "Vegetation": 9,
             "Vehicles": 10,
             "Wall": 11,
             "TrafficsSigns": 12,
             "Sky": 13,
             "Ground": 14,
             "Bridge": 15,
             "RailTrack": 16,
             "GuardRail": 17,
             "TrafficLight": 18,
             "Static": 19,
             "Dynamic": 20,
             "Water": 21,
             "Terrain": 22
             }

key_list = list(object_id.keys())
value_list = list(object_id.values())


def semantic_lidar_data(point_cloud_data):
    distance_name_data = {}

    object_list = dict()
    object_list['building'] = np.uint8([[[70, 70, 70]]])        
    object_list['pedestrian'] = np.uint8([[[220, 20, 60]]])
    object_list['vegetation'] = np.uint8([[[107, 142, 35]]])
    object_list['car'] = np.uint8([[[ 0, 0, 142]]])
    object_list['fence'] = np.uint8([[[ 190, 153, 153]]])
    object_list['traffic_sign'] = np.uint8([[[220, 220, 0]]])
    object_list['pole'] = np.uint8([[[153, 153, 153]]])
    object_list['wall'] = np.uint8([[[102, 102, 156]]])

    for detection in point_cloud_data:
        if detection.object_tag<=22:
            position = value_list.index(detection.object_tag)
            distance = math.sqrt((detection.point.x ** 2) + (detection.point.y ** 2) + (detection.point.z ** 2))
            distance_name_data["distance"] = distance
            distance_name_data["name"] = key_list[position]
            #write code here to display only name of object
            with open('output.txt', 'a') as file:
                file.write("Name of all objects nearby car  : - {}\n".format(distance_name_data['name']))
            if detection.object_tag == 5:
                with open ('traffic_sign.txt', 'a') as file:
                    file.write("Finally got one traffic_sign, distance is : {}\n".format(distance_name_data['distance']))
                image = image_queue.get()
                image_seg  = image_queue_seg.get()
                image.save_to_disk("traffic_sign/%06d.png" %(image.frame))
                image_seg.save_to_disk("traffic_sign_seg/%06d_semseg.png" %(image.frame), carla.ColorConverter.CityScapesPalette)
                img = cv2.imread("traffic_sign/%06d.png" % image.frame)
                img_semseg_bgr = cv2.imread("traffic_sign_seg/%06d_semseg.png" % image.frame)
                img_semseg_bgr = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGRA2BGR)
                img_semseg_hsv = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGR2HSV)
                mask = get_mask(img_semseg_hsv, object_list['traffic_sign'])
                bboxes = get_bbox_from_mask(mask)
                output_dir = "cropped_traffic"
                os.makedirs(output_dir, exist_ok=True)
                for i, bbox in enumerate(bboxes):
        
                    minr, minc, maxr, maxc = bbox
                    cv2.rectangle(img, (minc,minr), (maxc, maxr), (255,255,255), 6)#drawing the mask
                    cropped_image = img[minr:maxr, minc:maxc]
                    cropped_image_path = os.path.join(output_dir, f"{image.frame:06d}_cropped_{i}.png")
                    cv2.imwrite(cropped_image_path, cropped_image)
        else:
            continue


def process_img(image):
    i = np.array(image.raw_data)
    print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i3/255.0

def get_mask(seg_im, rgb_value):
    # rgb_value should be somethiing like np.uint8([[[70, 70, 70]]])
    # seg_im should be in HSV
    
    hsv_value = cv2.cvtColor(rgb_value, cv2.COLOR_RGB2HSV)
    
    hsv_low = np.array([[[hsv_value[0][0][0]-5, hsv_value[0][0][1], hsv_value[0][0][2]-5]]])
    hsv_high = np.array([[[hsv_value[0][0][0]+5, hsv_value[0][0][1], hsv_value[0][0][2]+5]]])
    
    mask = cv2.inRange(seg_im, hsv_low, hsv_high)
    return mask

def get_bbox_from_mask(mask):
    label_mask = measure.label(mask)
    props = measure.regionprops(label_mask)
    
    return [prop.bbox for prop in props]

def encode_if_bytes(obj):
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, dict):
        return {k: encode_if_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_if_bytes(v) for v in obj]
    return obj

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(200.0)
    world = client.load_world('Town06')
    env_obj = world.get_environment_objects()
    object_labels = [obj.type for obj in env_obj]
    unique_labels = set(object_labels)
    for label in unique_labels:
        print(label)

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05 #must be less than 0.1, or else physics will be noisy
    #must use fixed delta seconds and synchronous mode for python api controlled sim, or else 
    #camera and sensor data may not match simulation properly and will be noisy 
    world.apply_settings(settings)

    weather = carla.WeatherParameters(
        cloudiness=20.0,
        precipitation=20.0,
        sun_altitude_angle=110.0)

    #or use precomputed weathers
    #weather = carla.WeatherParameters.WetCloudySunset

    world.set_weather(weather)


    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter("model3")[0]

    # lets choose a random spawn point
    transform = random.choice(world.get_map().get_spawn_points()) 

    #spawn a vehicle
    vehicle = world.spawn_actor(bp, transform) 
    actor_list.append(vehicle)
    vehicle.set_autopilot(True)

    #lets create waypoints for driving the vehicle around automatically
    m= world.get_map()
    waypoint = m.get_waypoint(transform.location) #Unkown for now

    #lets add more vehicles
    #for _ in range(0, 5):
    #    transform = random.choice(m.get_spawn_points())

     #   bp_vehicle = random.choice(blueprint_library.filter('vehicle'))

        # This time we are using try_spawn_actor. If the spot is already
        # occupied by another object, the function will return None.
     #   other_vehicle = world.try_spawn_actor(bp_vehicle, transform)
      #  if other_vehicle is not None:
            #print(npc)
      #      other_vehicle.set_autopilot(True)
      #      actor_list.append(other_vehicle)

    # Adding random objects
    blueprint_library = world.get_blueprint_library()
    weirdobj_bp = blueprint_library.find('static.prop.fountain')
    weirdobj_transform = random.choice(world.get_map().get_spawn_points())
    weirdobj_transform = carla.Transform(carla.Location(x=230, y=195, z=40), carla.Rotation(yaw=180))
    weird_obj = world.try_spawn_actor(weirdobj_bp, weirdobj_transform)
    actor_list.append(weird_obj)

    #example for getting camera image
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")
    camera_transform = carla.Transform(carla.Location(x = -5.5, z = 2.5), carla.Rotation(pitch=8.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    actor_list.append(camera)

    #example for getting depth camera image
    camera_depth = blueprint_library.find('sensor.camera.depth')
    camera_depth.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_depth.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_depth.set_attribute("fov", "110")
    camera_transform = carla.Transform(carla.Location(x = -5.5, z = 2.5), carla.Rotation(pitch=8.0))
    camera_d = world.spawn_actor(camera_depth, camera_transform, attach_to=vehicle)
    image_queue_depth = queue.Queue()
    camera_d.listen(image_queue_depth.put)
    actor_list.append(camera_d)

    #example for getting semantic segmentation camera image
    camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_semseg.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_semseg.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_semseg.set_attribute("fov", "110")
    camera_transform = carla.Transform(carla.Location(x = -5.5, z = 2.5), carla.Rotation(pitch=8.0))
    camera_seg = world.spawn_actor(camera_semseg, camera_transform, attach_to=vehicle)
    image_queue_seg = queue.Queue()
    camera_seg.listen(image_queue_seg.put)
    actor_list.append(camera_seg)

    #Lidar sensor
    lidar_sensor = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    lidar_sensor.set_attribute('channels', str(64))
    lidar_sensor.set_attribute("points_per_second",str(56000))
    lidar_sensor.set_attribute("rotation_frequency",str(80))
    lidar_sensor.set_attribute("range",str(25))
    lidar_sensor.set_attribute("upper_fov", str(80))
    lidar_sensor.set_attribute("lower_fov", str(-80))
    sensor_lidar_spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2.0), carla.Rotation(pitch=0.000000, yaw=90.0, roll=0.000000))
    camera_l = world.spawn_actor(lidar_sensor, sensor_lidar_spawn_point, attach_to=vehicle)

    camera_l.listen(lambda point_cloud_data: semantic_lidar_data(point_cloud_data))
    actor_list.append(camera_l)

    #rgb camera
    image = image_queue.get()

    #semantic segmentation camera
    image_seg  = image_queue_seg.get()

    #depth camera
    image_depth = image_queue_depth.get()

    image.save_to_disk("test_images/%06d.png" %(image.frame))
    image_seg.save_to_disk("test_images/%06d_semseg.png" %(image.frame), carla.ColorConverter.CityScapesPalette)
    image_depth.save_to_disk("test_images/%06d_depth.png" %(image.frame), carla.ColorConverter.LogarithmicDepth)

    img = cv2.imread("test_images/%06d.png" % image.frame)
    img_semseg = mpimg.imread("test_images/%06d_semseg.png" % image.frame)
    img_depth = mpimg.imread("test_images/%06d_depth.png" % image.frame)


    img_semseg_bgr = cv2.imread("test_images/%06d_semseg.png" % image.frame)
    img_semseg_bgr = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGRA2BGR)
    img_semseg_hsv = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGR2HSV) # color wise segmentation is better in hsv space


    object_list = dict()
    object_list['building'] = np.uint8([[[70, 70, 70]]])        
    object_list['pedestrian'] = np.uint8([[[220, 20, 60]]])
    object_list['vegetation'] = np.uint8([[[107, 142, 35]]])
    object_list['car'] = np.uint8([[[ 0, 0, 142]]])
    object_list['fence'] = np.uint8([[[ 190, 153, 153]]])
    object_list['traffic_sign'] = np.uint8([[[220, 220, 0]]])
    object_list['pole'] = np.uint8([[[153, 153, 153]]])
    object_list['wall'] = np.uint8([[[102, 102, 156]]])

    
    
    
   
    mask = get_mask(img_semseg_hsv, object_list['traffic_sign'])
    #mask1 = get_mask(img_semseg_hsv, object_list['building'])
    #bboxes = get_bbox_from_mask(np.ma.mask_or(mask))
    bboxes = get_bbox_from_mask(mask)
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,18))
    #ax1.imshow(mask)

    output_dir = "cropped_images"
    os.makedirs(output_dir, exist_ok=True)
    for i, bbox in enumerate(bboxes):
        
        minr, minc, maxr, maxc = bbox
        cv2.rectangle(img, (minc,minr), (maxc, maxr), (255,255,255), 6)#drawing the mask
        cropped_image = img[minr:maxr, minc:maxc]
        cropped_image_path = os.path.join(output_dir, f"{image.frame:06d}_cropped_{i}.png")
        cv2.imwrite(cropped_image_path, cropped_image)


    #ax2.imshow(img)
    #ax2.imshow(cropped_image)
    #plt.show()

    weirdobj_loc = weird_obj.get_location()
    # returns x, y, z as weirdobj_loc.x, weirdobj_loc.y, weirdobj_loc.z

    weirdobj_transform = weird_obj.get_transform()
    # returns x, y, z as weirdobj_transform.location.x, weirdobj_transform.location.y, weirdobj_transform.location.z
    # also returns pitch, yaw, roll as weirdobj_transform.rotation.pitch, weirdobj_transform.rotation.yaw, weirdobj_transform.rotation.roll

    #similarly we can get the camera transform
    camera_transform = camera.get_transform()

    waypoint = random.choice(waypoint.next(1.5)) #navigate to next waypoint on map 1.5 meters ahead
    vehicle.set_transform(waypoint.transform) 
    dataset_dicts = []
    global_count=0

    for i in range(600):
        #rgb camera
        image = image_queue.get()

        #semantic segmentation camera
        image_seg  = image_queue_seg.get()
        #image_seg.convert(carla.ColorConverter.CityScapesPalette)

        #depth camera
        image_depth = image_queue_depth.get()
        #image_depth.convert(carla.ColorConverter.LogarithmicDepth)

        if i%30==0: #how many photos per frame?
            image.save_to_disk("test_images/%06d.png" %(image.frame))
            image_seg.save_to_disk("test_images/%06d_semseg.png" %(image.frame), carla.ColorConverter.CityScapesPalette)
            image_depth.save_to_disk("test_images/%06d_depth.png" %(image.frame), carla.ColorConverter.LogarithmicDepth)

            img = mpimg.imread("test_images/%06d.png" % image.frame)
            img_semseg = mpimg.imread("test_images/%06d_semseg.png" % image.frame)
            img_depth = mpimg.imread("test_images/%06d_depth.png" % image.frame)
        
            ## COCO format stuff, each image needs to have these keys
            height, width = cv2.imread("test_images/%06d.png" %(image.frame)).shape[:2]
            record = {}
            record['file_name'] = "test_images/%06d.png" %(image.frame)
            global_count+=1
            record['image_id'] = global_count
            record['height'] = height
            record['width'] = width
        
        
            #fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (12,18))
        
        
        
            ## compute bboxes from semseg
            img_semseg_bgr = cv2.imread("test_images/%06d_semseg.png" % image.frame)
            img_semseg_bgr = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGRA2BGR)
            img_semseg_hsv = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGR2HSV) # color wise segmentation is better in hsv space

            #bgr value exmaples of few objects: full list at https://carla.readthedocs.io/en/0.9.9/ref_sensors/ 
            object_list = dict()
            object_list['building'] = np.uint8([[[70, 70, 70]]])        
            object_list['pedestrian'] = np.uint8([[[220, 20, 60]]])
            object_list['vegetation'] = np.uint8([[[107, 142, 35]]])
            object_list['car'] = np.uint8([[[ 0, 0, 142]]])
            object_list['fence'] = np.uint8([[[ 190, 153, 153]]])
            object_list['traffic_sign'] = np.uint8([[[220, 220, 0]]])
            object_list['pole'] = np.uint8([[[153, 153, 153]]])
            object_list['wall'] = np.uint8([[[102, 102, 156]]])
        
            object_bboxes = dict()
            objects = []
            obj_id = 0
            obj2id = dict()
            for obj in object_list:
                mask = get_mask(img_semseg_hsv, object_list['traffic_sign'])
                #mask1 = get_mask(img_semseg_hsv, object_list['building'])
                #bboxes = get_bbox_from_mask(np.ma.mask_or(mask))
                bboxes = get_bbox_from_mask(mask)
                #fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,18))
                #ax1.imshow(mask)

                output_dir = "cropped_images"
                os.makedirs(output_dir, exist_ok=True)
                for i, bbox in enumerate(bboxes):
                    
                    minr, minc, maxr, maxc = bbox
                    cv2.rectangle(img, (minc,minr), (maxc, maxr), (255,255,255), 6)#drawing the mask
                    cropped_image = img[minr-2:maxr+2, minc-2:maxc+2]
                    cropped_image_path = os.path.join(output_dir, f"{image.frame:06d}_cropped_{i}.png")
                    cv2.imwrite(cropped_image_path, cropped_image)
            
                #let's visualize car bboxes
                if obj=='traffic_sign':
                    #ax4.imshow(mask)
                    for bbox in bboxes:
                        minr, minc, maxr, maxc = bbox
                        cv2.rectangle(img_semseg_bgr, (minc,minr), (maxc, maxr), (255,255,255), 6)
                        #ax5.imshow(img_semseg_bgr)
                    

            
                #lets put things in coco format for finetuning mask rcnn
                for bbox in bboxes:
                    minr, minc, maxr, maxc = bbox
                    obj_mask = np.copy(mask)
                    obj_mask[:minr] = 0
                    obj_mask[:, :minc] = 0
                    obj_mask[maxr+1:] = 0
                    obj_mask[:, maxc+1:] = 0

                    coco_rle_mask = pycocotools.mask.encode(np.array(obj_mask, order="F"))
                
                    obj_ann = {
                            'bbox': [minc, minr, maxc, maxr],
                            'bbox_mode': BoxMode.XYXY_ABS,
                            'segmentation': coco_rle_mask,
                            'category_id': obj_id
                    }
                    objects.append(obj_ann)
                
                    obj_id+=1
                    obj2id[obj] = obj_id
        
        

            record['annotations'] = objects
            record= encode_if_bytes(record)
        
            #print(record)
        
            dataset_dicts.append(record)
            
            
            #plt.show()
        
        #drive vehicle to next waypoint on map
        waypoint = random.choice(waypoint.next(1.5))
        vehicle.set_transform(waypoint.transform)
        


        # Apply this function to the entire dataset_dicts
        processed_dataset_dicts = [encode_if_bytes(record) for record in dataset_dicts]

        # Now, try saving it to a JSON file
        with open('dataset.json', 'w') as file:
            json.dump(processed_dataset_dicts, file)
        time.sleep(1000)

finally:
    #make sure to destroy all cameras and actors since they remain in the simulator even if you respawn using python. 
#It gets destroyed only if you restart CARLA simulator
    camera.destroy()
    camera_d.destroy()
    camera_seg.destroy()
    camera_l.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
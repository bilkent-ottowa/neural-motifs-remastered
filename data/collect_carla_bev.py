"""
Author : Yiğit Yıldırım
Last updated : 2024-11-2

Description:
This script is used to save the BEV images of the Carla dataset. We only need to save RGB images, bounding boxes, and class labels.
The class labels are the intersection of the classes in the VG dataset and the Carla dataset. The script saves the data in a .json file.

Change log:
    - 2024-11-2: Initial version of the script is created.
"""

import carla
import queue
import numpy as np
import json
import os
import cv2
import pickle

TAGS = {1: "street", # roads
        2: "sidewalk", # sidewalks
        3: "building", # buildings
        6: "pole",  # poles
        12: "person", # pedestrians
        14: "car",  # cars 
        15: "truck", # trucks
        16: "bus", # buses
        18: "motorcycle", # motorcycles
        19: "bike"} # bicycles
        #24: "sign"} # RoadLines (not sure)
                    # The tags are from the CARLA ObjectLabels.h



class CarlaBEVSampler:
    def __init__(self, client: carla.Client, tmPort = 8000, mode = 'sync', numBurn = 0, save_dir = 'data/carla_bev', save_name = 'carla_bev.json'):
        """
        CARLA BEV data sampler class
        :param client: carla.Client object, the client object to connect to the CARLA simulator.
        :param tmPort: int, the port number of the traffic manager (default is 8000).
        :param mode: str, the mode of the data sampler (default is 'sync').
        :param numBurn: int, the number of burn ticks (default is 0).
        :param save_dir: str, the directory to save the data (default is 'data/carla_bev').
        :param save_name: str, the name of the file to save the data (default is 'carla_bev.json').
        """
        
        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.bpLib = self.world.get_blueprint_library()
        self.tm = client.get_trafficmanager(tmPort)
        self.spawnPoints = self.map.get_spawn_points()

        assert mode in ['sync', 'async'], 'Invalid mode. Please use either sync or async.'

        self.set_mode(mode)

        self.save_dir = save_dir
        self.save_name = save_name
        self.bev_data = []

        # Create queue for the data
        self.RGB_que = queue.Queue()
        self.SEG_que = queue.Queue()

        # Spawn the actors
        carSP = self.spawnPoints[0]
        # carSP.location.x += 40
        rgbSP = carla.Transform(carla.Location(x=-3.5, y=25, z=60), carla.Rotation(pitch=-90))
        segSP = carla.Transform(carla.Location(x=-3.5, y=25, z=60), carla.Rotation(pitch=-90))

        # Create a RGB camera.
        self.rgb_cam = RGB_Cam(rgbSP, self.bpLib, self.world, callback=self.RGB_que.put)

        # Create a semantic segmentation camera.
        self.seg_cam = Semantic_Cam(segSP, self.bpLib, self.world, callback=self.SEG_que.put)

        # Create a car.
        self.car = ActorCar(self.bpLib, self.world, carSP)

        self.all_actors = [self.rgb_cam, self.seg_cam, self.car]
        self.timestamp = 0
        if self.mode == 'sync':
            for tick in range(numBurn):
                # Burn the first NUM_BURN_TICKS.
                self.world.tick()
                # self.timestamp += 0.05
                self.RGB_que.get()
                self.SEG_que.get()

        filedir = os.path.dirname(__file__)
        self.vgg_metadata = loadVG_SGG(os.path.join(filedir, 'stanford_filtered/VG-SGG-dicts.json'))
        self.vgg_labels = self.vgg_metadata['label_to_idx']

    def set_mode(self, mode):
        """
        Change the mode of the data sampler.
        :param mode: str, the mode of the data sampler (either 'sync' or 'async').
        """

        assert mode in ['sync', 'async'], 'Invalid mode. Please use either sync or async.'
        self.mode = mode

        currentSettings = self.world.get_settings()
        if self.mode == 'sync':
            currentSettings.synchronous_mode = True
            currentSettings.fixed_delta_seconds = 0.05
        else:
            currentSettings.synchronous_mode = False

        self.world.apply_settings(currentSettings)

        if self.mode == 'sync':
            self.world.tick()

        else:
            self.world.wait_for_tick()

    def rgb_seg_fuse(self):
        """
        Fuse the RGB and semantic segmentation images.
        :return: bounding_boxes, labels, rgb_im.
        """
        img_id = self.timestamp // 0.05

        rgb_im = self.RGB_que.get() # Get carla.Image object.
        seg_im = self.SEG_que.get() # Get carla.Image object.

        # Convert the carla.Image object to np.ndarray.
        rgb_array = np.frombuffer(rgb_im.raw_data, dtype=np.dtype("uint8"))
        rgb_array = np.reshape(rgb_array, (rgb_im.height, rgb_im.width, 4))
        # rgb_im = cv2.cvtColor(rgb_array, cv2.COLOR_BGRA2RGBA)
        
        seg_array = np.frombuffer(seg_im.raw_data, dtype=np.dtype("uint8"))
        seg_array = np.reshape(seg_array, (seg_im.height, seg_im.width, 4))
        seg_array = seg_array[:, :, :3]
        seg_array = seg_array[:, :, ::-1]
        
        # Start fusing
        detectedTags = seg_array[:, :, 0] # Tags are stored in the red channel.
        gt_bbs = []
        gt_classes = []

        for tag, name in TAGS.items():
            objectPx = np.where( detectedTags == tag)  # Get the pixels of the object.

            if len(objectPx[0]) == 0:
                continue
            objIds = np.unique(seg_array[objectPx][:, 1:], axis=0)  # Get the object ids.

            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0

            for objId in objIds:
                objLoc = np.where(np.all(seg_array[:,:,1:] == objId, axis=-1))  # Get the location of the object.
                y_min, x_min = np.min(objLoc, axis=1)
                y_max, x_max = np.max(objLoc, axis=1)
                gt_bb = [x_min, y_min, x_max, y_max]
                vgg_tag = self.vgg_labels[name]
                gt_class = [img_id, vgg_tag] 

                gt_bbs.append(gt_bb)
                gt_classes.append(gt_class)
        
        return img_id, np.asarray(gt_bbs), np.asarray(gt_classes), rgb_array
    
    def run(self, numTicks):
        """
        Run the data sampler.
        :param numTicks: int, the number of ticks to run the data sampler.
        """
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        img_path = os.path.join(self.save_dir, 'images')

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        for tick in range(numTicks):
            img_id, gt_bbs, gt_classes, rgb_im = self.rgb_seg_fuse()
            self.bev_data.append({'id': img_id, 'gt_bbs': gt_bbs, 'gt_classes': gt_classes})
            self.world.tick()
            self.timestamp += 0.05
            cv2.imwrite(os.path.join(img_path, f'{img_id}.png'), rgb_im)
            

    def save(self):
        """
        Save the data to a .json file.
        """
        with open(os.path.join(self.save_dir, self.save_name), 'wb') as f:
            pickle.dump(self.bev_data, f)



## HELPER CLASSES ##
class RGB_Cam:
    def __init__(self, 
                 position: carla.Transform,
                 bpLib,
                 world, 
                 Sx: int=800, 
                 Sy: int=600,
                 callback: callable=None):
        """
        DataSampler class is used to sample data from the CARLA simulator.
        Args:
            position: carla.Transform object, the position of the sensor.
            Sx: int, the width of the image.
            Sy: int, the height of the image.
            callback: callable, the callback function to process the image.
        """
        self.bp = bpLib.find('sensor.camera.rgb')
        self.bp.set_attribute('image_size_x', str(Sx))
        self.bp.set_attribute('image_size_y', str(Sy))

        self.cam = world.spawn_actor(self.bp, position)
        self.cam.listen(callback)

    def destroy(self):
        self.cam.destroy()

class Semantic_Cam:
    def __init__(self, 
                 position: carla.Transform, 
                 bpLib,
                 world,
                 Sx: int=800, 
                 Sy: int=600,
                 callback: callable=None):
        """
        DataSampler class is used to sample data from the CARLA simulator.
        Args:
            position: carla.Transform object, the position of the sensor.
            Sx: int, the width of the image.
            Sy: int, the height of the image.
            callback: callable, the callback function to process the image.
        """
        self.bp = bpLib.find('sensor.camera.instance_segmentation')
        self.bp.set_attribute('image_size_x', str(Sx))
        self.bp.set_attribute('image_size_y', str(Sy))

        self.cam = world.spawn_actor(self.bp, position)
        self.cam.listen(callback)

    def destroy(self):
        self.cam.destroy()

class ActorCar:
    def __init__(self,
                 bpLib,
                 world,
                 position: carla.Transform):
        """
        ActorCar class is used to spawn a car in the CARLA simulator, and drive it around.
        Args:
            position: carla.Transform object, the position of the car.
        """
        bp = bpLib.filter('vehicle.tesla.model3')[0]
        self.car = world.spawn_actor(bp, position)

        self.car.set_autopilot(True)
    def destroy(self):
        self.car.destroy()

### Helper functions ###
def loadVG_SGG(path):
    """
    Load the VG-SGG metadata.
    Args:
        path: str, the path to the VG-SGG metadata.
    Returns:
        metadata: dict, the metadata of the VG-SGG dataset.
    """

    with open(path, 'r') as f:
        metadata = json.load(f)
    
    return metadata
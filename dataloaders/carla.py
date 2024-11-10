"""
Author : Yiğit Yıldırım
Last updated : 2024-11-2

Description:
This script is used to retrieve saved bounding boxes and class labels. The implementation is based on the PyTorch framework, and
follows VG dataload format of neural-motifs. The script loads the saved data and returns the bounding boxes and class labels.

Change log:
    - 2024-11-2: Initial version of the script is created.
"""
import os
import json
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
# from dataloaders.blob import Blob

from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop


class CarlaBEV(Dataset):
    def __init__(self, mode, carla_file, metadata_file, 
                 im_folder, filter_empty_rels=True, num_im=-1, num_val_im=5000, filter_duplicate_rels=True,
                 filter_non_overlap=True, use_proposals=False):
        
        """
        Dataset for the CarlaBEV dataset.
        :param mode: (str) 'train', 'val', or 'test'
        :param carla_file: (str) Path to the CarlaBEV file
        :param metadata_file: (str) Path to the metadata file
        :param im_folder: (str) Path to the folder containing images
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals        
        """

        assert mode in ['train', 'val', 'test'], "Mode must be in 'train', 'val', 'test'"

        self.mode = mode

        # Initialize the dataset
        self.carla_file = carla_file
        self.metadata_file = metadata_file
        self.im_folder = im_folder
        self.filter_empty_rels = filter_empty_rels
        self.filter_duplicate_rels = filter_duplicate_rels and mode == 'train'

        self.ind_to_classes, self.ind_to_predicates = self.load_metadata(metadata_file)
        self.carla = self.load_carla(carla_file)
        
        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

        self.filenames = [f for f in os.listdir(im_folder) if f.endswith('.png')]
        self.gt_boxes = [self.carla[i]['boxes'] for i in range(len(self.carla))]
        self.gt_classes = [self.carla[i]['gt_classes'] for i in range(len(self.carla))]

    def load_metadata(self, path):
        with open(path, 'r') as f:
            metadata = json.load(f)

        metadata['label_to_idx']['__background__'] = 0
        metadata['predicate_to_idx']['__background__'] = 0

        class_to_ind = metadata['label_to_idx']
        predicate_to_ind = metadata['predicate_to_idx']

        ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
        ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

        return ind_to_classes, ind_to_predicates
    
    def load_carla(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __getitem__(self, index):
        imFile = os.path.join(self.im_folder, f'{index}.png')
        image_unpadded = Image.open(imFile).convert('RGB')

        gt_boxes = [self.carla[i]['boxes'] for i in range(len(self.carla)) if self.carla[i]['id'] == index]
        gt_classes = [self.carla[i]['gt_classes'] for i in range(len(self.carla)) if self.carla[i]['id'] == index]

        # Boxes are already at BOX_SCALE
        if self.is_train:
            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1])
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0])

            # # crop the image for data augmentation
            # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

        w, h = image_unpadded.size
        box_scale_factor = BOX_SCALE / max(w, h)

        if False: # Flip the image if training. Hardcoded to false for now.
            scaled_w = int(box_scale_factor * float(w))
            # print("Scaled w is {}".format(scaled_w))
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = IM_SCALE / max(w, h)
        if h > w:
            im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': self.gt_classes[index].copy(),
            'gt_relations': None,
            'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
            'index': index,
            'flipped': False,
            'fn': self.filenames[index],
        }

        assertion_checks(entry)

        return entry
    
    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)
    

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()

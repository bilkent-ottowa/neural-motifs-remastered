from dataloaders.carla import CarlaBEV
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE, CARLA_IMAGES, CARLA_DATA_PATH, VG_SGG_DICT_FN
import dill as pkl
import os
from torch.utils.data import DataLoader
from dataloaders.carla import vg_collate



conf = ModelConfig()

carlaData = CarlaBEV(mode = 'test',
                     carla_file = CARLA_DATA_PATH, # Fix paths
                     metadata_file = VG_SGG_DICT_FN,
                     im_folder = CARLA_IMAGES,
                     filter_empty_rels = True,
                     filter_duplicate_rels = True,
                     num_im = -1,
                     num_val_im = 5000,
                     filter_non_overlap = True,
                     use_proposals = False)


# carlaDataLoader = DataLoader(carlaData, batch_size = conf.batch_size, shuffle = False, num_workers = conf.num_workers,
#                              collate_fn=lambda x: vg_collate(x, 2,False, 'rel'), drop_last = True)

carlaDataLoader = DataLoader(carlaData, batch_size = conf.batch_size, shuffle = False, num_workers = conf.num_workers,
                             drop_last = True)

for i, data in enumerate(carlaDataLoader):
    ims = data['img']
    gt_boxes = data['gt_boxes']
    gt_classes = data['gt_classes']
    im_sizes = data['img_size']
    # image_offset = data['image_offset']

    print(ims.shape)
    print(gt_boxes.shape)
    print(gt_classes.shape)
    print(im_sizes)
    # print(image_offset)


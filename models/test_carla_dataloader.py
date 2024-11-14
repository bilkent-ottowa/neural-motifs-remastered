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


carlaDataLoader = DataLoader(carlaData, batch_size = conf.batch_size, shuffle = False, num_workers = conf.num_workers,
                             collate_fn=lambda x: vg_collate(x, 2,False, 'rel'), drop_last = True)

for val_b, batch in enumerate(tqdm(carlaDataLoader)):
    print(len(batch)) # Should be 6
    im0 = batch[0]
    print(f'Image 0 shape: {im0[0].shape}, expected {3}x{IM_SCALE}x{IM_SCALE}') # Should be (batch_size, 3, 592, 592)
    print(f'Image 0 im sizes: {im0[1]}') # Should be (batch_size, 3)
    print(f'Image 0 image offsets: {im0[2]}') # Should be (0)
    print(f'Image 0 gt boxes: {im0[3].shape}') # Should be (num_gt_boxes, 4)
    print(f'Image 0 gt classes: {im0[4].shape}') # Should be (num_gt_boxes, 2)
    

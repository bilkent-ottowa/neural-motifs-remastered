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

conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

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

carlaDataLoader = DataLoader(carlaData, batch_size = conf.batch_size, shuffle = True, num_workers = conf.num_workers)

detector = RelModel(classes=carlaData.ind_to_classes, rel_classes=carlaData.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision
                    )

detector.cuda()
ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])
all_pred_entries = []

def val_batch(batch_num, b, thrs=(20, 50, 100)):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):

        assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
        # assert np.all(rels_i[:,2] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)

detector.eval()
for val_b, batch in enumerate(tqdm(carlaDataLoader)):
    val_batch(conf.num_gpus*val_b, batch)

if conf.cache is not None:
    with open(conf.cache,'wb') as f:
        pkl.dump(all_pred_entries, f)


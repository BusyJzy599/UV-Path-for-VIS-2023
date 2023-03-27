from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil

import torch
from torch.nn import Parameter

def save_checkpoint(state, is_best, fpath,logger):
    #
    if is_best:
        # shutil.copy(fpath, osp.join(osp.dirname(fpath), 'o2u_model_best.pth.tar'))
        torch.save(state, fpath)
        logger.info("Save best trained model, auc score:%f"%state['best_auc'])

def load_checkpoint(fpath,logger):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        logger.info("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

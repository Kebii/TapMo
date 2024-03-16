import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        if self.model.input_feats_rot == (self.njoints+1) * 6:
            self.nfeats = 9
        elif self.model.input_feats_rot == (self.njoints+1) * 4:
            self.nfeats = 7
        else:
            self.nfeats = 7
        self.cond_mode = self.model.cond_mode

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out, modify = self.model(x, timesteps, y)
        out_uncond, modify_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond)), modify


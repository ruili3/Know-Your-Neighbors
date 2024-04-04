from collections import OrderedDict

from torch import profiler

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import os
import torch.nn.functional as F
from models.common.model.layers import *


class VLFuser(nn.Module):
    def __init__(self,
                v_feat_ch,
                vl_feat_ch,
                fuse_type,
                hidden_dim=256):
        super().__init__()
        self.v_feat_ch = v_feat_ch
        self.vl_feat_ch = vl_feat_ch
        self.fuse_type = fuse_type
        self.hidden_dim = hidden_dim

        # expand vision feat dim and add with vlseg feat
        if self.fuse_type == "add":
            self.out_ch = self.vl_feat_ch
            self.vision_proc = nn.Sequential(nn.Conv2d(self.v_feat_ch, self.out_ch, 1),
                                                nn.ReLU(),
                                                nn.Conv2d(self.vl_feat_ch, self.out_ch, 1)
                                            )
        # direct concat and output
        elif self.fuse_type == "direct_concat":
            self.out_ch = self.vl_feat_ch + self.v_feat_ch
        # direct concat and squeeze dims
        elif self.fuse_type =="concat_reg":
            self.fused_reg = nn.Sequential(nn.Conv2d(self.vl_feat_ch + self.v_feat_ch, self.hidden_dim, 1),
                                                nn.ReLU(),
                                                nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
                                            )
            self.out_ch = self.hidden_dim
        elif self.fuse_type =="vision_feat_only":
            self.out_ch = self.v_feat_ch

    @property
    def latent_size(self):
        return self.out_ch

    def forward(self, vision_feat, vl_feat, seg_mask, text_feats, seg_logits):

        # extract the last feature map from feature lists 
        vision_feat = vision_feat[0]

        if self.fuse_type == "add":
            return self.vision_proc(vision_feat) + vl_feat
        elif self.fuse_type == "direct_concat":
            return torch.cat((vision_feat, vl_feat), dim=1)
        elif self.fuse_type =="concat_reg":
            return self.fused_reg(torch.cat((vision_feat, vl_feat), dim=1))
        elif self.fuse_type =="vision_feat_only":
            return vision_feat

    @classmethod
    def from_conf(cls, conf, **kwargs):
        assert  kwargs.get("vl_feat_ch", None) is not None
        return cls(
            v_feat_ch=conf.get("vision_d_out"),
            vl_feat_ch = kwargs.get("vl_feat_ch", None),
            fuse_type = conf.get("fuse_type"),
            hidden_dim = 256
        )
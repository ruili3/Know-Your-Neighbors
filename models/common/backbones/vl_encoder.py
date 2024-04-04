from collections import OrderedDict

from torch import profiler

from models.common.model.layers import *
from models.common.backbones.monodepth2 import *
from models.lseg.lseg_net import LSegFeatNet
from models.common.model.vl_fuser import *


import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import os
import torch.nn.functional as F


def get_labels(path):
    all_labels = []
    with open(os.path.join(path)) as ff:
        lines = ff.readlines()
        for line in lines:
            label = line.strip()
            all_labels.append(label)
    return all_labels


def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)


def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad


class VisionLanguageEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        vision_encoder,
        vlseg_encoder,
        fuser,
        vl_model_cropsize,
        freeze_vlseg_model,
        return_only_feat,
        vlseg_model_name
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.vlseg_encoder = vlseg_encoder
        if self.vlseg_encoder is not None:
            self.vlseg_encoder.eval()

        self.fuser = fuser
        self.vl_model_cropsize = vl_model_cropsize
        self.freeze_vlseg_model = freeze_vlseg_model
        self.return_only_feat = return_only_feat

        self.vlseg_model_name = vlseg_model_name



    def single_crop_forward(self, image, crop_size):

        batch, _, h, w = image.size()

        assert w > h # below code is only valid when w > h
        height = int(crop_size * h / w)
        # resize image to current size
        cur_img = resize_image(image, height, crop_size, mode="bilinear")

        pad_img = pad_image(cur_img, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5], crop_size=crop_size)

        out, image_feats, text_feats = self.vlseg_encoder(pad_img)

        out = resize_image(crop_image(out, 0, height, 0, crop_size), h, w, mode="bilinear")
        image_feats = resize_image(crop_image(image_feats, 0, height, 0, crop_size), h, w, mode="bilinear")

        return out, image_feats, text_feats



    def single_crop_forward_dpt(self, image, crop_size):

        batch, _, h, w = image.size()

        assert w > h # below code is only valid when w > h
        height = int(crop_size * h / w)
        # resize image to current size
        cur_img = resize_image(image, height, crop_size, mode="bilinear")

        pad_img = pad_image(cur_img, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5], crop_size=crop_size)

        out, image_feats = self.vlseg_encoder(pad_img)

        out = resize_image(crop_image(out, 0, height, 0, crop_size), h, w, mode="bilinear")
        
        assert isinstance(image_feats, tuple)

        updated_img_feat = []
        for level, feat in enumerate(image_feats):
            feat = resize_image(feat, crop_size, crop_size, mode="bilinear")
            feat = crop_image(feat, 0, height, 0, crop_size)
            feat = resize_image(feat, h // 2 ** (level+1), w // 2 ** (level+1), mode="bilinear")
            updated_img_feat.append(feat)

        return out, updated_img_feat



    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W), normalized to [-1, 1]
        :return latent (B, latent_size, H, W)
        """

        vision_feat = None
        seg_mask = None
        text_feat = None
        vl_feat = None
        seg_logits = None


        # follow BTS
        if self.vision_encoder is not None:
            vision_feat = self.vision_encoder(x)

        # vlseg takes normalized image
        if self.vlseg_encoder is not None:
            if self.vlseg_model_name == "lseg":
                if self.freeze_vlseg_model == "freeze_all" or self.freeze_vlseg_model is True:
                    self.vlseg_encoder.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            seg_logits, vl_feat, text_feat = self.single_crop_forward(x, self.vl_model_cropsize)
                elif self.freeze_vlseg_model == "freeze_backbone":
                    self.vlseg_encoder.pretrained.eval()
                    self.vlseg_encoder.clip_pretrained.eval()
                    seg_logits, vl_feat, text_feat = self.single_crop_forward(x, self.vl_model_cropsize)
                else:
                    raise NotImplementedError()
            
            else:
                raise NotImplementedError()


        seg_mask = torch.max(seg_logits, 1)[1].unsqueeze(1)

        fused_feat = self.fuser(vision_feat, vl_feat, seg_mask, text_feat, seg_logits)

        if self.return_only_feat:
            return fused_feat
        else:
            return {
                    "fused_feat": fused_feat,
                    "text_feat": text_feat,
                    "seg_logits": seg_logits,
                    "seg_mask": seg_mask
            }



    @property
    def latent_size(self):
        return self.fuser.latent_size

    @property
    def text_embedding_ch(self):
        return self.vlseg_encoder.get_out_ch



    @classmethod
    def from_conf(cls, conf):
        
        use_vision_model = conf.get("use_vision_model")
        use_vlseg_model = conf.get("use_vlseg_model")
        assert (use_vision_model or use_vlseg_model)
        vl_model_cropsize = 480

        label_set = get_labels(conf.get("ov_label_path"))
        
        vlseg_model_name = conf.get("vlseg_model_name", "lseg")

        # vision model
        vision_model_name = conf.get("vl_visionmodel_name", "Monodepth2")
        vision_encoder = globals()[vision_model_name](resnet_layers=conf.get("resnet_layers"),
                                        cp_location=conf.get("vision_cp_location"),
                                        freeze=conf.get("freeze_vision_model"),
                                        num_ch_dec=conf.get("num_ch_dec"),
                                        d_out=conf.get("vision_d_out"),
                                        scales=range(4)) if use_vision_model else None
        
        # VL model
        if vlseg_model_name == "lseg":
            vlseg_encoder = LSegFeatNet(path=conf.get("lseg_model_path"),
                                        labels=label_set,
                                        backbone="clip_vitl16_384",
                                        features=256,
                                        crop_size=vl_model_cropsize,
                                        arch_option=0,
                                        block_depth=0,
                                        activation='lrelu')


        if not use_vlseg_model:
            vlseg_encoder = None

        if vlseg_encoder is not None:
            vlseg_encoder.pretrained.model.patch_embed.img_size = (vl_model_cropsize, vl_model_cropsize)
            if conf.get("freeze_vlseg_model", "freeze_all") == "freeze_all" or conf.get("freeze_vlseg_model", "freeze_all") is True:
                for p in vlseg_encoder.parameters():
                    p.requires_grad = False
            elif conf.get("freeze_vlseg_model", "freeze_all") == "freeze_backbone":
                for p in vlseg_encoder.clip_pretrained.parameters():
                    p.requires_grad = False
                for p in vlseg_encoder.pretrained.parameters():
                    p.requires_grad = False
            else:
                raise NotImplementedError()

        # Fuser Model
        fuser_name = conf.get("vl_fuser_name", "VLFuser")
        if vlseg_model_name == "lseg":
            vlfeat_ch = vlseg_encoder.get_out_ch
            
        fuser = globals()[fuser_name].from_conf(conf,
                                                vl_feat_ch = vlfeat_ch,
                                                num_ch_enc = vision_encoder.encoder.num_ch_enc,
                                                )


        return cls(
            vision_encoder = vision_encoder,
            vlseg_encoder = vlseg_encoder,
            fuser = fuser,
            vl_model_cropsize = vl_model_cropsize,
            freeze_vlseg_model = conf.get("freeze_vlseg_model"),
            return_only_feat = conf.get("return_only_feat"),
            vlseg_model_name = vlseg_model_name
        )


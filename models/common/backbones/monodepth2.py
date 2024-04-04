"""
Implements image encoders
"""

from collections import OrderedDict

from torch import profiler

from models.common.model.layers import *

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models
import torch.utils.model_zoo as model_zoo


# Code taken from https://github.com/nianticlabs/monodepth2
#
# Godard, Clément, et al.
# "Digging into self-supervised monocular depth estimation."
# Proceedings of the IEEE/CVF international conference on computer vision.
# 2019.

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            # x = self.convs[("upconv", i, 0)](x)
            x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                if x[0].shape[2] > input_features[i - 1].shape[2]:
                    x[0] = x[0][:, :, :input_features[i - 1].shape[2], :]
                if x[0].shape[3] > input_features[i - 1].shape[3]:
                    x[0] = x[0][:, :, :, :input_features[i - 1].shape[3]]
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            #x = self.convs[("upconv", i, 1)](x)
            x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

            self.outputs[("features", i)] = x

            if i in self.scales:
                #self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("disp", i)] = self.sigmoid(self.decoder[self.decoder_keys[("dispconv", i)]](x))

        return self.outputs


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec=None, d_out=1, scales=range(4), use_skips=True):
        super(Decoder, self).__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        if num_ch_dec is None:
            self.num_ch_dec = np.array([128, 128, 256, 256, 512])
        else:
            self.num_ch_dec = num_ch_dec
        self.d_out = d_out
        self.scales = scales

        self.num_ch_dec = [max(self.d_out, chns) for chns in self.num_ch_dec]

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.d_out)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        with profiler.record_function("encoder_forward"):
            self.outputs = {}

            # decoder
            x = input_features[-1]
            for i in range(4, -1, -1):
                x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)

                x = [F.interpolate(x, scale_factor=(2, 2), mode="nearest")]

                if self.use_skips and i > 0:
                    feats = input_features[i - 1]

                    # NOTE in the default setting, it is not necessary since the resolution is the same
                    if x[0].shape[2] > feats.shape[2]:
                        x[0] = x[0][:, :, :feats.shape[2], :]
                    if x[0].shape[3] > feats.shape[3]:
                        x[0] = x[0][:, :, :, :feats.shape[3]]
                    x += [feats]
                x = torch.cat(x, 1)

                x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

                self.outputs[("features", i)] = x

                if i in self.scales:
                    self.outputs[("disp", i)] = self.decoder[self.decoder_keys[("dispconv", i)]](x)

        return self.outputs


class Monodepth2(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        resnet_layers=18,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # decoder
        self.decoder = Decoder(num_ch_enc=self.num_ch_enc, d_out=self.d_out, num_ch_dec=num_ch_dec, scales=self.scales)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
            outputs = self.decoder(image_features)
            # the `disp` here refers to the feature channels
            x = [outputs[("disp", i)] for i in self.scales]

        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers=conf.get("resnet_layers", 18)
        )



class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ap', pretrained=pretrained)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

        self.num_ch_enc = np.array([24, 40, 64, 176, 2048])

    def forward(self, x):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for ii in range(3):
            x[:,ii,:,:] = (x[:,ii,:,:] - mean[ii]) / std[ii]
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))

        return (features[4], features[5], features[6], features[8], features[11])

class EfficientNetAE(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4),
        resnet_layers = None
    ):
        super().__init__()

        self.encoder = EfficientNetEncoder()
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # decoder
        self.decoder = Decoder(num_ch_enc=self.num_ch_enc, d_out=self.d_out, num_ch_dec=num_ch_dec, scales=self.scales)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
            outputs = self.decoder(image_features)
            # the `disp` here refers to the feature channels
            x = [outputs[("disp", i)] for i in self.scales]

        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers = None
        )


class AttnDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec=None, d_out=1, scales=range(4), use_skips=True):
        super(AttnDecoder, self).__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        if num_ch_dec is None:
            self.num_ch_dec = np.array([128, 128, 256, 256, 512])
        else:
            self.num_ch_dec = num_ch_dec
        self.d_out = d_out
        self.scales = scales

        self.num_ch_dec = [max(self.d_out, chns) for chns in self.num_ch_dec]

        # decoder
        self.convs = OrderedDict()

        latent_ch = [64, 64, 128, 256, 512]

        # feature fusion
        self.convs["f4"] = Attention_Module(self.num_ch_enc[4]  , latent_ch[4])
        self.convs["f3"] = Attention_Module(self.num_ch_enc[3]  , latent_ch[3])
        self.convs["f2"] = Attention_Module(self.num_ch_enc[2]  , latent_ch[2])
        self.convs["f1"] = Attention_Module(self.num_ch_enc[1]  , latent_ch[1])
        
        # adapt input feat_ch to the desired input_ch
        self.is_transform = False
        if num_ch_enc[0] != latent_ch[0]:
            self.is_transform = True
            self.convs["f0"] = ConvBlock(self.num_ch_enc[0], latent_ch[0])
        self.num_ch_enc = latent_ch

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.d_out)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        with profiler.record_function("encoder_forward"):
            self.outputs = {}
            feat=[]
            feat.append(self.decoder[self.decoder_keys["f0"]](input_features[0]) if self.is_transform else input_features[0])
            feat.append(self.decoder[self.decoder_keys["f1"]](input_features[1]))
            feat.append(self.decoder[self.decoder_keys["f2"]](input_features[2]))
            feat.append(self.decoder[self.decoder_keys["f3"]](input_features[3]))
            feat.append(self.decoder[self.decoder_keys["f4"]](input_features[4]))
                        
            # decoder
            x = feat[-1]
            for i in range(4, -1, -1):
                x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)

                x = [F.interpolate(x, scale_factor=(2, 2), mode="nearest")]

                if self.use_skips and i > 0:
                    feats = feat[i - 1]

                    # NOTE in the default setting, it is not necessary since the resolution is the same
                    if x[0].shape[2] > feats.shape[2]:
                        x[0] = x[0][:, :, :feats.shape[2], :]
                    if x[0].shape[3] > feats.shape[3]:
                        x[0] = x[0][:, :, :, :feats.shape[3]]
                    x += [feats]
                x = torch.cat(x, 1)

                x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

                self.outputs[("features", i)] = x

                if i in self.scales:
                    self.outputs[("disp", i)] = self.decoder[self.decoder_keys[("dispconv", i)]](x)

        return self.outputs




class Mono2Attn(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        resnet_layers=18,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # decoder
        self.decoder = AttnDecoder(num_ch_enc=self.num_ch_enc, 
                                    d_out=self.d_out, 
                                    num_ch_dec=num_ch_dec, 
                                    scales=self.scales)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
            outputs = self.decoder(image_features)
            # the `disp` here refers to the feature channels
            x = [outputs[("disp", i)] for i in self.scales]

        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers=conf.get("resnet_layers", 18)
        )


class Mono2Enc(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        resnet_layers=18,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.d_out = d_out
        # self.scales = scales
        # decoder
        # self.decoder = Decoder(num_ch_enc=self.num_ch_enc, d_out=self.d_out, num_ch_dec=num_ch_dec, scales=self.scales)
        # self.num_ch_dec = self.decoder.num_ch_dec
        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
        return image_features

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers=conf.get("resnet_layers", 18)
        )




class HRDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec=None, d_out=1, scales=range(4), use_skips=True):
        super(HRDecoder, self).__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        if num_ch_dec is None:
            self.num_ch_dec = np.array([128, 128, 256, 256, 512])
        else:
            self.num_ch_dec = num_ch_dec
        
        self.d_out = d_out
        self.num_ch_dec = [max(self.d_out, chns) for chns in self.num_ch_dec]

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                if j==0:
                    num_ch_in = self.num_ch_enc[i]
                else:
                    num_ch_in = self.num_ch_dec[i+1]
                
                num_ch_out = self.num_ch_dec[i]
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(self.num_ch_dec[row+1], self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row+1] +
                                                                        self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(self.num_ch_dec[row+1] + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])


        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.d_out)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.decoder[self.decoder_keys["X_" + index + "_attention"]](
                    self.decoder[self.decoder_keys["X_{}{}_Conv_0".format(row+1, col-1)]](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.decoder[self.decoder_keys["X_{}{}_Conv_0".format(row + 1, col - 1)]],
                        self.decoder[self.decoder_keys["X_{}{}_Conv_1".format(row + 1, col - 1)]]]
                if col != 1:
                    conv.append(self.decoder[self.decoder_keys["X_" + index + "_downsample"]])
                                                                    
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.decoder[self.decoder_keys["X_04_Conv_0"]](x)
        x = self.decoder[self.decoder_keys["X_04_Conv_1"]](upsample(x))
        outputs[("disp", 0)] = self.decoder[self.decoder_keys["dispConvScale0"]](x)
        outputs[("disp", 1)] = self.decoder[self.decoder_keys["dispConvScale1"]](features["X_04"])
        outputs[("disp", 2)] = self.decoder[self.decoder_keys["dispConvScale2"]](features["X_13"])
        outputs[("disp", 3)] = self.decoder[self.decoder_keys["dispConvScale3"]](features["X_22"])
        return outputs




class MonoViTDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec=None, d_out=1, scales=range(4)):
        super(MonoViTDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256]) if num_ch_dec is None else num_ch_dec
        self.convs = nn.ModuleDict()
        
        latent_ch = [64, 64, 128, 256, 512]
        self.d_out = d_out


        # feature fusion
        self.convs["f4"] = Attention_Module(self.num_ch_enc[4]  , latent_ch[4])
        self.convs["f3"] = Attention_Module(self.num_ch_enc[3]  , latent_ch[3])
        self.convs["f2"] = Attention_Module(self.num_ch_enc[2]  , latent_ch[2])
        self.convs["f1"] = Attention_Module(self.num_ch_enc[1]  , latent_ch[1])
        
        # adapt input feat_ch to the desired input_ch
        self.is_transform = False
        if num_ch_enc[0] != latent_ch[0]:
            self.is_transform = True
            self.convs["f0"] = ConvBlock(self.num_ch_enc[0], latent_ch[0])

            
        self.num_ch_dec = [max(self.d_out, chns) for chns in self.num_ch_dec]
        self.num_ch_enc = latent_ch



        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        for j in range(5):
            for i in range(5 - j):
                if j==0:
                    num_ch_in = self.num_ch_enc[i]
                else:
                    num_ch_in = self.num_ch_dec[i+1]
                num_ch_out = self.num_ch_dec[i]
                
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(self.num_ch_dec[row+1], self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row+1] +
                                                                        self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(self.num_ch_dec[row+1] + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.d_out)
                

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        feat={}
        feat[4] = self.decoder[self.decoder_keys["f4"]](input_features[4])
        feat[3] = self.decoder[self.decoder_keys["f3"]](input_features[3])
        feat[2] = self.decoder[self.decoder_keys["f2"]](input_features[2])
        feat[1] = self.decoder[self.decoder_keys["f1"]](input_features[1])
        # adapt input feat_ch to the desired input_ch
        feat[0] = self.decoder[self.decoder_keys["f0"]](input_features[0]) if self.is_transform else input_features[0]
        
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.decoder[self.decoder_keys["X_" + index + "_attention"]](
                    self.decoder[self.decoder_keys["X_{}{}_Conv_0".format(row+1, col-1)]](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.decoder[self.decoder_keys["X_{}{}_Conv_0".format(row + 1, col - 1)]],
                        self.decoder[self.decoder_keys["X_{}{}_Conv_1".format(row + 1, col - 1)]]]
                if col != 1:
                    conv.append(self.decoder[self.decoder_keys["X_" + index + "_downsample"]])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.decoder[self.decoder_keys["X_04_Conv_0"]](x)
        x = self.decoder[self.decoder_keys["X_04_Conv_1"]](upsample(x))
        outputs[("disp", 0)] = self.decoder[self.decoder_keys["dispconv0"]](x)
        outputs[("disp", 1)] = self.decoder[self.decoder_keys["dispconv1"]](features["X_04"])
        outputs[("disp", 2)] = self.decoder[self.decoder_keys["dispconv2"]](features["X_13"])
        outputs[("disp", 3)] = self.decoder[self.decoder_keys["dispconv3"]](features["X_22"])

        return outputs
        

class Mono2HRDecoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        resnet_layers=18,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # decoder MonoViTDecoder HRDecoder
        self.decoder = HRDecoder(num_ch_enc=self.num_ch_enc, 
                                        d_out=self.d_out, 
                                        num_ch_dec=num_ch_dec, 
                                        scales=self.scales)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
            outputs = self.decoder(image_features)
            # the `disp` here refers to the feature channels
            x = [outputs[("disp", i)] for i in self.scales]

        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers=conf.get("resnet_layers", 18)
        )



class Mono2MViTDecoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        resnet_layers=18,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # decoder MonoViTDecoder HRDecoder
        self.decoder = MonoViTDecoder(num_ch_enc=self.num_ch_enc, 
                                        d_out=self.d_out, 
                                        num_ch_dec=num_ch_dec, 
                                        scales=self.scales)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
            outputs = self.decoder(image_features)
            # the `disp` here refers to the feature channels
            x = [outputs[("disp", i)] for i in self.scales]

        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers=conf.get("resnet_layers", 18)
        )

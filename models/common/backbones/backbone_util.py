from models.common.backbones.image_encoder import ImageEncoder
from models.common.backbones.monodepth2 import Monodepth2, Mono2Enc, Mono2HRDecoder, Mono2MViTDecoder, Mono2Attn
from models.common.backbones.spatial_encoder import SpatialEncoder
from models.common.backbones.vl_encoder import VisionLanguageEncoder


def make_backbone(conf, **kwargs):
    enc_type = conf.get("type", "monodepth2")  # spatial | global
    if enc_type == "monodepth2":
        net = Monodepth2.from_conf(conf, **kwargs)
    elif enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    elif enc_type == "vl_encoder":
        net = VisionLanguageEncoder.from_conf(conf, **kwargs)
    elif enc_type == "mono2hr":
        net = Mono2HRDecoder.from_conf(conf, **kwargs)
    elif enc_type == "mono2mvit":
        net = Mono2MViTDecoder.from_conf(conf, **kwargs)
    elif enc_type == "mono2attn":
        net = Mono2Attn.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported encoder type: {enc_type}")
    return net

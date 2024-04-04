from .mlp import ImplicitNet
from .resnetfc import ResnetFC
from .vl_mod_attention import VLModAttention

def make_mlp(conf, d_in, d_latent=0, d_text=0, allow_empty=False, **kwargs):
    mlp_type = conf.get("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "vl_modulation_attention":
        net = VLModAttention.from_conf(conf, d_in, d_text, **kwargs)
    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net

from torch import nn
import torch
import torch.autograd.profiler as profiler
from models.common import util
from models.common.model.resnetfc import ResnetBlockFC
import numpy as np
import torch.nn.functional as F
from models.common.model.linear_attn import MHLinearAttention
import math



def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class VLModAttention(nn.Module):
    def __init__(
        self,
        d_in,
        text_d_in,
        d_out=1,
        d_hidden=128,
        attn_head = 4,
        attn_kv_ch = None,
        use_q_residual = True,
        lin_attn_type = "img_softmax_q",
        use_valid_pts_mask = True,
        blocks = 4,
        skip_layers = [2],
        return_q_feat = False,
        use_pe_3D = False
    ):
        """
        This class enriches the input visual features with rich semantics using VL modulation operations,
        then aggreagtes spatial context via attention operation.

        [Parameters]
        d_in: dimension of the input visual features (appearance encoder and VL-image encoder)
        text_d_in: dimension of the text features
        d_out: output dimension of the estimated density
        d_hidden: hidden dimension in the modulation and attention operations
        attn_head: number of attention heads
        blocks: total number of vl-modulation layers
        skip_layers: modulation layers with residual connection
        use_pe_3D: specify if using additional 3D positional embeddings
        """
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.use_valid_pts_mask = use_valid_pts_mask
        self.text_d_in = text_d_in
        self.skip_layers = skip_layers
        self.blocks = blocks
        self.activation = nn.ReLU()
        self.return_q_feat = return_q_feat
        self.use_pe_3D = use_pe_3D

        if self.use_pe_3D:
            self.pe3D = PositionalEmbedding3D(d_hidden)

        # input text encoding
        self.lin_in_text = nn.Linear(text_d_in, d_hidden)
        nn.init.constant_(self.lin_in_text.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_in_text.weight, a=0, mode="fan_in")

        # VL-modulation layers
        for i in range(blocks):
            if i == 0:
                layer = nn.Linear(d_in, d_hidden)
            elif i in self.skip_layers:
                layer = nn.Linear(d_in + d_hidden, d_hidden)
            else:
                layer = nn.Linear(d_hidden, d_hidden)
            # initial weights
            layer.apply(weights_init)
            setattr(self, f"mlp_encoding_{i+1}", layer)

        # vl-spatial attention
        self.attn_kv_ch = self.d_hidden // attn_head if attn_kv_ch is None else attn_kv_ch
        self.point_attn = MHLinearAttention(n_head=attn_head, 
                                                d_model=self.d_hidden, 
                                                d_k=self.attn_kv_ch, 
                                                d_v=self.attn_kv_ch,
                                                use_q_residual=use_q_residual,
                                                lin_attn_type=lin_attn_type)

        self.lin_out = nn.Linear(self.d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")


    def forward(self, zx, text_x, combine_inner_dims=(1,), combine_index=None, dim_size=None, view_dirs=None, mask=None, **kwargs):

        text_bias = self.lin_in_text(text_x.float())
        x = zx

        # VL-Modulation
        for i in range(self.blocks):
            x = getattr(self, f"mlp_encoding_{i+1}")(x)
            x = x * text_bias
            x = self.activation(x)
            if (i+1) in self.skip_layers:
                x = torch.cat([zx, x], dim=-1)

        validmask = mask if self.use_valid_pts_mask else None
        # add 3D pe
        if self.use_pe_3D and kwargs.get("xyz", None) is not None:
            xyz_coord = kwargs.get("xyz", None)
            x = x + self.pe3D(xyz_coord)
        
        # VL-spatial attention
        x = self.point_attn(x, text_bias, x, validmask)
        out = self.lin_out(x)

        if self.return_q_feat:
            return out, x
        else:
            return out


    @classmethod
    def from_conf(cls, conf, d_in, text_d_in, **kwargs):
        return cls(
            d_in,
            text_d_in,
            d_hidden = conf.get("d_hidden", 128),
            attn_head = conf.get("attn_head", 4),
            attn_kv_ch = conf.get("attn_kv_ch", None),
            lin_attn_type = conf.get("lin_attn_type", "img_softmax_q"),
            use_q_residual = conf.get("use_q_residual", True),
            use_valid_pts_mask = conf.get("use_valid_pts_mask", True),
            blocks = conf.get("n_blocks", 4),
            skip_layers = conf.get("skip_layers", [2]),
            return_q_feat=conf.get("return_q_feat", False),
            use_pe_3D = conf.get("use_pe_3D", False),
            **kwargs
        )




class PositionalEmbedding3D(nn.Module):
    def __init__(self, feat_ch):
        super(PositionalEmbedding3D, self).__init__()
        self.feat_ch = feat_ch

    def forward(self, point_pos):
        # point_pos is of shape (B, n_pts, 3)
        B, n_pts, _ = point_pos.size()

        # Positional encoding
        x = point_pos[:, :, 0].unsqueeze(2)  # B, n_pts, 1
        y = point_pos[:, :, 1].unsqueeze(2)  # B, n_pts, 1
        z = point_pos[:, :, 2].unsqueeze(2)  # B, n_pts, 1

        pos_enc = torch.cat([torch.sin(x), torch.cos(x),
                             torch.sin(y), torch.cos(y),
                             torch.sin(z), torch.cos(z)], dim=2)  # B, n_pts, 6

        # Linear layer to match feat_ch
        linear_layer = nn.Linear(6, self.feat_ch).to(point_pos.device)
        pos_enc = linear_layer(pos_enc)  # B, n_pts, feat_ch

        return pos_enc
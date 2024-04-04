import torch
from torch import nn, einsum


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def exists(val):
    return val is not None


def linear_attn_img(q, k, v, norm_queries=True, kv_mask = None):
    """
    input shape should be: bs, head, len, dim 
    mask shape: bs, len
    """
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        # print("k:{}, mask:{}".format(k.shape, mask.shape))
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q * dim ** -0.5
    k = k * dim ** -0.5

    if norm_queries:
        q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)


def linear_attn_nlp(q, k, v, kv_mask = None):
    """
    input shape should be: bs, head, len, dim 
    mask shape: bs, len
    """
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    q = q * dim ** -0.5

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)


class MHLinearAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, use_q_residual=True, dropout=0.1,
                lin_attn_type="img_q", fc_dim_type="same"):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        if fc_dim_type == "same":
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        elif fc_dim_type == "kv":
            self.fc = nn.Linear(n_head * d_v, n_head * d_v, bias=False)
            self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)
            assert use_q_residual == False

        self.use_q_residual = use_q_residual
        self.lin_attn_type = lin_attn_type

        assert lin_attn_type in ["img_softmax_q", "img_no_softmax_q", "nlp"]

        # self.dropout = nn.Dropout(dropout)
        

    def forward(self, q, k, v, mask=None):
        """
        q: bs, len, hidden_dim
        mask shape: bs, len
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        if self.use_q_residual:
            residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).permute(0, 2, 1, 3) # bs, head, len, dim
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).permute(0, 2, 1, 3)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).permute(0, 2, 1, 3)

        if self.lin_attn_type == "img_softmax_q":
            q = linear_attn_img(q, k, v, norm_queries=True, kv_mask=mask)
        elif self.lin_attn_type == "img_no_softmax_q":
            q = linear_attn_img(q, k, v, norm_queries=False, kv_mask=mask)
        elif self.lin_attn_type == "nlp":
            q = linear_attn_nlp(q, k, v, kv_mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        q = self.fc(q)
        if self.use_q_residual:
            q += residual
        q = self.layer_norm(q)

        return q



'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on timm code base
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
 插入可变窗口注意力机制 VSAWindowAttention
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from einops import rearrange
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import _cfg, PatchEmbed, resize_pos_embed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    MLP用于Vision Transformer, MLP- mixer及相关网络
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        '''
        init函数用于在类实例化时初始化对象，in_features：输入特征维度，hidden_features：隐藏层的维度（默认为None，若不指定则使用in_features），
        out_features：输出特征的维度（默认值为 None，如果不指定则使用 in_features），act_layer：激活函数层，默认使用 nn.GELU，
        drop：dropout的概率，默认值为 0，即没有 dropout
        '''
        super().__init__()  # 调用父类的构造函数，初始化父类
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 定义第一个全连接层（fc1），将维度in_features映射到维度hidden_features
        self.act = act_layer()  # 定义激活层（act），使用传入的激活函数类实例化一个激活函数对象，默认是 GELU 激活函数。
        self.fc2 = nn.Linear(hidden_features, out_features)  # 定义第二个全连接层（fc2），它将hidden_features映射到维度out_features
        self.drop = nn.Dropout(drop)  # 定义 dropout 层（drop），在训练过程中以drop的概率随机将某些神经元的输出设为0，以防止过拟合

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Attention(nn.Module):
#     # MultiHeadAttention 多头自注意力模块
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         # dim：输入维度，num_heads：多头注意力头数，qk_scale：缩放因子,
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads  # head_dim表示每个头的维度
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5  # 如果没有提供缩放因子，则使用head_dim的倒数平方根作为缩放因子
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         # 一个线性变换层，将输入特征映射到query、key、value的空间。它的输出维度是 dim * 3，因为我们需要为每个头生成 Q、K 和 V
#         self.attn_drop = nn.Dropout(attn_drop)
#         # self.attn_drop 是一个 dropout 层，用于在计算注意力权重时随机丢弃一些信息，以防止过拟合。
#         self.proj = nn.Linear(dim, dim)
#         # self.proj 是一个线性变换层，用于将多头注意力的输出映射回原始维度 dim
#         self.proj_drop = nn.Dropout(proj_drop)
#         # self.proj_drop 是一个 dropout 层，用于在投影过程中随机丢弃一些信息，以防止过拟合
#         self.attn_gradients = None
#         self.attention_map = None
#         # self.attn_gradients 和 self.attention_map 是用于保存注意力梯度和注意力图的变量，通常用于调试和可视化
#
#     def save_attn_gradients(self, attn_gradients):
#         self.attn_gradients = attn_gradients
#
#     def get_attn_gradients(self):
#         return self.attn_gradients
#
#     def save_attention_map(self, attention_map):
#         self.attention_map = attention_map
#
#     def get_attention_map(self):
#         return self.attention_map
#
#     def forward(self, x, register_hook=False):
#         B, N, C = x.shape   # 获取输入张量 x 的形状信息，B 表示批量大小，N 表示序列长度，C 表示特征维度
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         # 计算注意力分数，首先计算Q和K的乘积，然后进行缩放。接着应用 softmax 函数得到注意力权重，并通过 dropout 层处理以减少过拟合
#
#         if register_hook:
#             self.save_attention_map(attn)
#             attn.register_hook(self.save_attn_gradients)
#             # 如果 register_hook 为 True，表示要记录注意力权重和梯度。则调用 save_attention_map 方法记录注意力权重，
#             # 并调用 save_attn_gradients 方法注册钩子以保存注意力梯度
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         # 使用注意力权重对值进行加权求和，然后将结果重排并重新形状为 (B, N, C)。
#         # 接着通过投影层 self.proj 进行线性变换，并通过 dropout 层处理以减少过拟合
#         return x


# class Attention(nn.Module):
#     # 可变窗口注意力 VSWAttention
#     def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., img_size=(1,1), out_dim=None, window_size=1):
#         super().__init__()
#         self.img_size = to_2tuple(img_size)
#         self.num_heads = num_heads
#         self.dim = dim
#         self.out_dim = out_dim or dim
#         self.relative_pos_embedding = True
#         head_dim = dim // self.num_heads
#         self.ws = window_size
#         self.shift_size = 0
#
#         self.padding_bottom = (self.ws - self.img_size[0] % self.ws) % self.ws
#         self.padding_right = (self.ws - self.img_size[1] % self.ws) % self.ws
#
#         self.sampling_offsets = nn.Sequential(
#             nn.AvgPool2d(kernel_size=window_size, stride=window_size),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
#         )
#         self.sampling_scales = nn.Sequential(
#             nn.AvgPool2d(kernel_size=window_size, stride=window_size),
#             nn.LeakyReLU(),
#             nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
#         )
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Conv2d(dim, out_dim * 3, 1, bias=qkv_bias)
#         # self.kv = nn.Conv2d(dim, dim*2, 1, bias=False)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(out_dim, out_dim, 1)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         if self.relative_pos_embedding:
#             # define a parameter table of relative position bias
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((window_size + window_size - 1) * (window_size + window_size - 1),
#                             num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#             # get pair-wise relative position index for each token inside the window
#             coords_h = torch.arange(self.ws)
#             coords_w = torch.arange(self.ws)
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#             coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#             relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
#             relative_coords[:, :, 1] += self.ws - 1
#             relative_coords[:, :, 0] *= 2 * self.ws - 1
#             relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#             self.register_buffer("relative_position_index", relative_position_index)
#
#             trunc_normal_(self.relative_position_bias_table, std=.02)
#             print('The relative_pos_embedding is used')
#
#         h, w = self.img_size
#         h, w = h + self.shift_size + self.padding_bottom, w + self.shift_size + self.padding_right
#         image_reference_w = torch.linspace(-1, 1, w)
#         image_reference_h = torch.linspace(-1, 1, h)
#         image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2,
#                                                                                                        1).unsqueeze(
#             0)  # 2, h, w
#         window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
#         window_num_h, window_num_w = window_reference.shape[-2:]
#         window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)
#
#         base_coords_h = torch.arange(self.ws) * 2 * self.ws / self.ws / (h - 1)
#         base_coords_h = (base_coords_h - base_coords_h.mean())
#         base_coords_w = torch.arange(self.ws) * 2 * self.ws / self.ws / (w - 1)
#         base_coords_w = (base_coords_w - base_coords_w.mean())
#
#         expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
#         assert expanded_base_coords_h.shape[0] == window_num_h
#         assert expanded_base_coords_h.shape[1] == self.ws
#         expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
#         assert expanded_base_coords_w.shape[0] == window_num_w
#         assert expanded_base_coords_w.shape[1] == self.ws
#         expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
#         expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
#         coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
#                                                                                                         1).reshape(1, 2,
#                                                                                                                    window_num_h,
#                                                                                                                    self.ws,
#                                                                                                                    window_num_w,
#                                                                                                                    self.ws)
#         self.base_coords = (window_reference + coords).cuda()
#         self.coords = coords.cuda()
#         # self.register_buffer('base_coords', window_reference+coords)
#         # self.register_buffer('coords', coords)
#
#     def forward(self, x, register_hook=False):
#         # print("_____x_____", x)
#         b, _, h, w = x.shape
#         shortcut = x
#         assert h == self.img_size[0]
#         assert w == self.img_size[1]
#
#         x = torch.nn.functional.pad(x, (self.shift_size, self.padding_right, self.shift_size, self.padding_bottom))
#         window_num_h, window_num_w = self.base_coords.shape[-4], self.base_coords.shape[-2]
#
#         coords = self.base_coords.repeat(b * self.num_heads, 1, 1, 1, 1, 1)
#         sampling_offsets = self.sampling_offsets(x)
#         num_predict_total = b * self.num_heads
#         sampling_offsets = sampling_offsets.reshape(num_predict_total, 2, window_num_h, window_num_w)
#         sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (w // self.ws)
#         sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (h // self.ws)
#
#         sampling_scales = self.sampling_scales(x)  # B, heads*2, h // window_size, w // window_size
#         sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)
#
#         coords = coords + self.coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :,
#                                                                                   None]
#         sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, self.ws * window_num_h,
#                                                                  self.ws * window_num_w, 2)
#
#         qkv = self.qkv(shortcut).reshape(b, 3, self.num_heads, self.out_dim // self.num_heads, h, w).transpose(1,
#                                                                                                                0).reshape(
#             3 * b * self.num_heads, self.out_dim // self.num_heads, h, w)
#         qkv = torch.nn.functional.pad(qkv, (
#         self.shift_size, self.padding_right, self.shift_size, self.padding_bottom)).reshape(3, b * self.num_heads,
#                                                                                             self.out_dim // self.num_heads,
#                                                                                             h + self.shift_size + self.padding_bottom,
#                                                                                             w + self.shift_size + self.padding_right)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         k_selected = F.grid_sample(
#             k.reshape(num_predict_total, self.out_dim // self.num_heads, h + self.shift_size + self.padding_bottom,
#                       w + self.shift_size + self.padding_right),
#             grid=sample_coords, padding_mode='zeros', align_corners=True
#         ).reshape(b * self.num_heads, self.out_dim // self.num_heads, h + self.shift_size + self.padding_bottom,
#                   w + self.shift_size + self.padding_right)
#         v_selected = F.grid_sample(
#             v.reshape(num_predict_total, self.out_dim // self.num_heads, h + self.shift_size + self.padding_bottom,
#                       w + self.shift_size + self.padding_right),
#             grid=sample_coords, padding_mode='zeros', align_corners=True
#         ).reshape(b * self.num_heads, self.out_dim // self.num_heads, h + self.shift_size + self.padding_bottom,
#                   w + self.shift_size + self.padding_right)
#
#         q = q.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.ws, window_num_w,
#                       self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w, self.num_heads,
#                                                                     self.ws * self.ws, self.out_dim // self.num_heads)
#         k = k_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.ws, window_num_w,
#                                self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w,
#                                                                              self.num_heads, self.ws * self.ws,
#                                                                              self.out_dim // self.num_heads)
#         v = v_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.ws, window_num_w,
#                                self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w,
#                                                                              self.num_heads, self.ws * self.ws,
#                                                                              self.out_dim // self.num_heads)
#
#         dots = (q @ k.transpose(-2, -1)) * self.scale
#
#         if self.relative_pos_embedding:
#             relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             dots += relative_position_bias.unsqueeze(0)
#
#         attn = dots.softmax(dim=-1)
#         x = attn @ v
#
#         x = rearrange(x, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
#         x = x[:, :, self.shift_size:h + self.shift_size, self.shift_size:w + self.shift_size]
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
#
#     def _reset_parameters(self):
#         nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
#         nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
#         nn.init.constant_(self.sampling_scales[-1].weight, 0.)
#         nn.init.constant_(self.sampling_scales[-1].bias, 0.)
#
#     def flops(self, ):
#         N = self.ws * self.ws
#         M = self.ws * self.ws
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * M
#         #  x = (attn @ v)
#         flops += self.num_heads * N * M * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         h, w = self.img_size[0] + self.shift_size + self.padding_bottom, self.img_size[
#             1] + self.shift_size + self.padding_right
#         flops *= (h / self.ws * w / self.ws)
#
#         # for sampling
#         flops_sampling = 0
#         # pooling
#         flops_sampling += h * w * self.dim
#         # regressing the shift and scale
#         flops_sampling += 2 * (h / self.ws + w / self.ws) * self.num_heads * 2 * self.dim
#         # calculating the coords
#         flops_sampling += h / self.ws * self.ws * w / self.ws * self.ws * 2
#         # grid sampling attended features
#         flops_sampling += h / self.ws * self.ws * w / self.ws * self.ws * self.dim
#
#         flops += flops_sampling
#
#         return flops

class Attention(nn.Module):
    # 可变窗口注意力 VSWAttention
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., window_size=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.ws = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_position_bias = nn.Parameter(torch.zeros((2 * window_size - 1)**2, num_heads))
        self.register_buffer("relative_position_index", self.calculate_relative_position_index(window_size))

    def calculate_relative_position_index(self, ws):
        # 计算相对位置索引
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij'), 0)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def forward(self, x, register_hook=False):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Decompose the combined tensor into q, k, v

        # Attention mechanism
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn += self.relative_position_bias[self.relative_position_index.view(-1)].view(H * W, H * W, -1).permute(2, 0, 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Projection and output
        x = (attn @ v).reshape(B, self.num_heads, C // self.num_heads, H, W).sum(dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # VSWAttention
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=7)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 use_grad_checkpointing=False, ckpt_layer=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i >= depth - ckpt_layer)
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)

        return x

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    #     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
    #         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
    #         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    #     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint

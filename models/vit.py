'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on timm code base
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
 2024.6.11修改
 1、添加Block类init中的use_entmax: bool = False, learnable_entmax_alpha: bool = False两个参数
 2、添加对象的实例化 self.SMHAttn
 3、添加SMHAttn的实现 x = x + self.SMHAttn(x)
'''

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed, resize_pos_embed
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from models.SparseAttention.sparse_mha import SparseMultiHeadSelfAttention


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


class Attention(nn.Module):
    # MultiHeadAttention 多头自注意力模块
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        # dim：输入维度，num_heads：多头注意力头数，qk_scale：缩放因子,
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # head_dim表示每个头的维度
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5  # 如果没有提供缩放因子，则使用head_dim的倒数平方根作为缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 一个线性变换层，将输入特征映射到query、key、value的空间。它的输出维度是 dim * 3，因为我们需要为每个头生成 Q、K 和 V
        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_drop 是一个 dropout 层，用于在计算注意力权重时随机丢弃一些信息，以防止过拟合。
        self.proj = nn.Linear(dim, dim)
        # self.proj 是一个线性变换层，用于将多头注意力的输出映射回原始维度 dim
        self.proj_drop = nn.Dropout(proj_drop)
        # self.proj_drop 是一个 dropout 层，用于在投影过程中随机丢弃一些信息，以防止过拟合
        self.attn_gradients = None
        self.attention_map = None
        # self.attn_gradients 和 self.attention_map 是用于保存注意力梯度和注意力图的变量，通常用于调试和可视化

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = x.shape  # 获取输入张量 x 的形状信息，B 表示批量大小，N 表示序列长度，C 表示特征维度
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 计算注意力分数，首先计算Q和K的乘积，然后进行缩放。接着应用 softmax 函数得到注意力权重，并通过 dropout 层处理以减少过拟合

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)
            # 如果 register_hook 为 True，表示要记录注意力权重和梯度。则调用 save_attention_map 方法记录注意力权重，
            # 并调用 save_attn_gradients 方法注册钩子以保存注意力梯度

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # 使用注意力权重对值进行加权求和，然后将结果重排并重新形状为 (B, N, C)。
        # 接着通过投影层 self.proj 进行线性变换，并通过 dropout 层处理以减少过拟合
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False,
                 use_entmax: bool = False, learnable_entmax_alpha: bool = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.SMHAttn = SparseMultiHeadSelfAttention(
            num_heads=num_heads,
            embed_dim=dim,
            dropout=drop,
            use_entmax=use_entmax,
            learnable_entmax_alpha=learnable_entmax_alpha)
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
        x = x + self.SMHAttn(self.norm2(x))
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
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 若未传入归一化函数则采用LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # img_size，in_chans规格的图像转成patch_size块维度为embed_dim的嵌入序列

        num_patches = self.patch_embed.num_patches  # 获取patch_embed中的块数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 创建一个可训练的全零张量作为cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # 创建一个可训练的全零张量作为位置嵌入
        self.pos_drop = nn.Dropout(p=drop_rate)  # 创建Drop层，以概率p将输入张量的部分元素置零

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule随机深度衰减规则
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i >= depth - ckpt_layer)
            )
            for i in range(depth)])  # 定义Block的实例，每个 Block 实例代表 Transformer 模型中的一个编码器层
        self.norm = norm_layer(embed_dim)  # 按输入维度embed_dim创建norm_layer归一化层

        trunc_normal_(self.pos_embed, std=.02)  # 使用std=.02的标准差,截断正态分布来初始化位置嵌入（pos_embed）
        trunc_normal_(self.cls_token, std=.02)  # 使用std=.02的标准差,截断正态分布来初始化cls令牌嵌入（cls_token）
        self.apply(self._init_weights)  # 递归调用self._init_weights函数，来初始化模型的权重

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    '''
    用于初始化模型的权重和归一化层的参数:
    如果输入的模块 m 是线性层 (nn.Linear)，则使用截断正态分布初始化权重 (m.weight)，标准差为 0.02。
    同时，如果该线性层有偏置项 (bias)，则将其初始化为零。
    如果输入的模块 m 是归一化层 (nn.LayerNorm)，则将其偏置项初始化为零，将其缩放参数初始化为 1.0。
    '''

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}  # 返回一个字典，其中包含不需要进行权重衰减的参数的名称

    def forward(self, x, register_blk=-1):
        # B = x.shape[0]  # 提取x的0号位作为批量大小
        B, C, H, W = x.shape
        print(x.shape)
        x = self.patch_embed(x)  # 调用patch_embed对象将输入x转为嵌入序列
        print("patch_embed(x)", x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # 扩展cls令牌的第一个维度为B，-1表示保持原维度不变
        x = torch.cat((cls_tokens, x), dim=1)  # 将cls_tokens与输入张量x在维度1上进行拼接，即在序列维度上进行拼接
        print("cat", x.shape)

        x = x + self.pos_embed[:, :x.size(1), :]  # 将位置嵌入pos_embed直接加到输入张量x上
        print("pos_embed", x.shape)

        x = self.pos_drop(x)  # 调用pos_drop层，将x部分元素置零
        print("pos_drop", x.shape)

        # 至此，得到了一个带有cls令牌和位置嵌入的输入张量x
        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)  # 对每个块应用一些个性化的操作
        x = self.norm(x)  # 对x进行归一化操作
        print("norm", x.shape)
        return x

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)  # 加载权重


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """
    Load weights from .npz checkpoints for official Google Brain Flax implementation
    从.npz检查点加载Google Brain flex官方实现的权重
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
    # interpolate position embedding 插入点嵌入
    # 实现了对预训练的位置嵌入进行插值操作，以适应新模型结构的变化
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

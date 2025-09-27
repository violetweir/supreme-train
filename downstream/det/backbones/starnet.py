import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import matplotlib
from functools import partial
from timm.models.layers import trunc_normal_
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.registry import MODELS
 

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.GELU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
    


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device="cuda")
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device="cuda")
    # dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device="cuda")
    # dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device="cuda")
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type, device="cuda").flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type, device="cuda").flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def create_learnable_wavelet_filter (in_size, out_size, filter_size=4, type=torch.float):
    """
    创建可学习的小波滤波器
    
    Args:
        in_size: 输入通道数
        out_size: 输出通道数
        filter_size: 滤波器大小
        type: 数据类型
    
    Returns:
        learnable_dec_filters: 可学习的分解滤波器
        learnable_rec_filters: 可学习的重构滤波器
    """

    # 初始化可学习滤波器参数
    # 使用正态分布初始化，模拟小波滤波器的特性
    dec_lo = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    dec_hi = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    rec_lo = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    rec_hi = torch.randn(filter_size, dtype=type, device="cuda") * 0.1
    
    # 将参数注册为可学习参数
    dec_lo = torch.nn.Parameter(dec_lo)
    dec_hi = torch.nn.Parameter(dec_hi)
    rec_lo = torch.nn.Parameter(rec_lo)
    rec_hi = torch.nn.Parameter(rec_hi)
    
    # 构建滤波器组
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape

    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)

    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape

    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)

    return x

class WTAttn(nn.Module):
    def __init__(self, dim, wt_type='db1', learnable_wavelet=False,stage=0):
        super(WTAttn, self).__init__()
        self.learnable_wavelet = learnable_wavelet
        
        if learnable_wavelet:
            # 使用可学习的小波滤波器
            wt_filter, iwt_filter = create_learnable_wavelet_filter(dim, dim, type=torch.float)
            # 将滤波器参数注册为模型参数
            self.wt_filter = nn.Parameter(wt_filter)
            self.iwt_filter = nn.Parameter(iwt_filter)
            self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
            self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        else:
            # 使用固定的小波滤波器
            wt_filter, iwt_filter = create_wavelet_filter(wt_type, dim, dim, torch.float)
            self.wt_function = partial(wavelet_transform, filters=wt_filter)
            self.iwt_function = partial(inverse_wavelet_transform, filters=iwt_filter)

        self.lh_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.hl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        if stage == 0:
            self.ll_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        elif stage == 1:
            self.ll_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        else :
            self.ll_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # self.attn_weight_linear = Conv2d_BN(dim,dim,ks=1)
        # self.attn_linear = Conv2d_BN(dim,dim,ks=1)
        self.act = nn.Hardsigmoid() # 或者使用 nn.Sigmoid()
        # self.act = EffectiveSELayer(dim)
        # self.ese = EffectiveSELayer(dim)

    

    
    def forward(self, x):


        x_wt = self.wt_function(x)
        ll, lh, hl, hh = x_wt[:, :, 0, :, :], x_wt[:, :, 1, :, :], x_wt[:, :, 2, :, :], x_wt[:, :, 3, :, :]

        
        # Apply convolutions
        lh_conv = self.lh_conv(lh)
        hl_conv = self.hl_conv(hl)
        ll_conv = self.ll_conv(ll)
        
        # Combine results
        # attn = self.ese(self.attn_linear(self.attn_weight_linear(self.act(lh_conv * hl_conv) * ll_conv))+ll)
        attn = (self.act((lh_conv * hl_conv)) * ll_conv )+ ll
        wt_map = torch.cat([attn.unsqueeze(2), lh_conv.unsqueeze(2), hl_conv.unsqueeze(2), hh.unsqueeze(2)], dim=2)
        output = self.iwt_function(wt_map)

        return output


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid()  # Using Hardshrink as PyTorch equivalent to Hardsigmoid

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

class Block(nn.Module):
    def __init__(self, dim, ffn_ratio, drop_path=0., wt_type='db1', learnable_wavelet=False,stage=0):
        super().__init__()
        self.DW = RepVGGDW(dim)
        self.ese = EffectiveSELayer(dim)
        self.ffn1 = Residual(FFN(dim, int(dim*ffn_ratio)))
        self.wtattn = Residual(WTAttn(dim, wt_type=wt_type, learnable_wavelet=learnable_wavelet))
        self.ffn2 = Residual(FFN(dim, int(dim * ffn_ratio)))
    
    def forward(self, x):
        x_shape = x.shape
        if (x_shape[2] % 2 > 0) or (x_shape[3] % 2 > 0):
            x_pads = (0, x_shape[3] % 2, 0, x_shape[2] % 2)
            x = F.pad(x, x_pads)
        x = self.DW(x)
        x = self.ese(x)
        x = self.ffn1(x)
        x = self.wtattn(x)
        x = self.ffn2(x)
        return x


@MODELS.register_module()
class StarNet(nn.Module):
    def __init__(self, 
                img_size=224, 
                in_chans=3, 
                num_classes=1000, 
                dims=[40,80,160,320], 
                depth=[1,2,4,5], 
                mlp_ratio=2., 
                act_layer="GELU", 
                drop_path_rate=0., 
                distillation=False, 
                head_init_scale=0. ,
                layer_scale_init_value=0., 
                learnable_wavelet=True,
                down_sample=32,
                sync_bn=False, out_indices=(1,2,3), pretrained=None, frozen_stages=-1, norm_eval=False):
        super().__init__()
        self.sync_bn = sync_bn
        self.out_indices = out_indices
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        if down_sample == 32:
            self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, dims[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 1, 1), torch.nn.ReLU(),
                                Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
                           )
        elif down_sample == 64:
            self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, dims[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(dims[0] // 4, dims[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(dims[0] // 2, dims[0], 3, 2, 1)
                           )
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        for i, (dim, dpth) in enumerate(
                            zip(dims,depth)):
            for j in range(dpth):
                blocks[i].append(Block(dim,ffn_ratio=mlp_ratio, wt_type='db1', learnable_wavelet=learnable_wavelet,stage=i))
            
            if i != len(depth) - 1:
                blk = blocks[i+1]
                blk.append(Conv2d_BN(dims[i], dims[i], ks=3, stride=2, pad=1, groups=dims[i]))
                blk.append(Conv2d_BN(dims[i], dims[i+1], ks=1, stride=1, pad=0))
        self._init_weights()

    def _init_weights(self):

        if self.pretrained is not None:

            self_state_dict = self.state_dict()
            original_weight = self_state_dict['blocks4.4.ffn2.m.pw1.bn.weight']
            print("*"*50)
            print("original weight:",original_weight)
            state_dict = torch.load(self.pretrained, map_location='cpu')
            for k, v in state_dict.items():
                if k in self_state_dict.keys():
                   self_state_dict.update({k: v})
            self.load_state_dict(self_state_dict, strict=True)
            self_state_dict = self.state_dict()
            new_weight = self_state_dict['blocks4.4.ffn2.m.pw1.bn.weight']
            # new_weight = self_state_dict['blocks4.4.ffn2.m.pw1.bn.weight']
            print("*"*50)
            print("new weight:",new_weight)
            print("*"*50)
            print(f'original weight mean: {original_weight.mean()}, std: {original_weight.std()}')
            print(f'load ckpt from {self.pretrained}')
            print('self pretrained is not None, model weights loaded from pretrained checkpoint.')


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'alpha', 'gamma', 'beta'}

    @torch.jit.ignore
    def no_ft_keywords(self):
        # return {'head.weight', 'head.bias'}
        return {}

    @torch.jit.ignore
    def ft_head_keywords(self):
        return {'head.weight', 'head.bias'}, self.num_classes

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()

    def check_bn(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
                m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)


    def forward(self, x):
        out = []
        x = self.patch_embed(x)
        x = self.blocks1(x)
        out.append(x)
        x = self.blocks2(x)
        out.append(x)
        x = self.blocks3(x)
        out.append(x)

        x = self.blocks4(x)
        out.append(x)

        out = tuple([out[i] for i in self.out_indices])
        return out
    
    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            m = getattr(self, f'blocks{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(StarNet, self).train(mode)
        # self._freeze_stages()
        # for name, param in self.blocks1.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(name)

        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()




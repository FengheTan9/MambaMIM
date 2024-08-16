import torch
import torch.nn as nn
from timm.models.layers import DropPath

_cur_active: torch.Tensor = None  # B1fff


# todo: try to use `gather` for speed?
def _get_active_ex_or_ii(H, W, D, returning_active_ex=True):
    h_repeat, w_repeat, d_repeat = H // _cur_active.shape[-3], W // _cur_active.shape[-2], D // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3).repeat_interleave(d_repeat, dim=4)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=True)  # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)

    bhwdc = x.permute(0, 2, 3, 4, 1)
    nc = bhwdc[ii]  # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)  # use BN1d to normalize this flatten feature `nc`

    bchwd = torch.zeros_like(bhwdc)
    bchwd[ii] = nc
    bchwd = bchwd.permute(0, 4, 1, 2, 3)
    return bchwd


def sp_in_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)
    bhwdc = x.permute(0, 2, 3, 4, 1)
    cn = bhwdc[ii].permute(1,
                           0)  # select the features on non-masked positions to form a flatten feature `nc` [17787, 3]
    C, N = cn.shape
    bcl = cn.reshape(C, -1, x.shape[0]).permute(2, 0, 1)
    bcl = super(type(self), self).forward(bcl)  # use BN1d to normalize this flatten feature `nc`
    nc = bcl.permute(1, 2, 0).reshape(C, -1).permute(1, 0)
    bchwd = torch.zeros_like(bhwdc)
    bchwd[ii] = nc
    bchwd = bchwd.permute(0, 4, 1, 2, 3)
    return bchwd


class SparseConv3d(nn.Conv3d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool3d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool3d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm3d(nn.BatchNorm1d):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm3d(nn.SyncBatchNorm):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseInstanceNorm3d(nn.InstanceNorm1d):
    forward = sp_in_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 5:  # BHWDC or BCHWD
            if self.data_format == "channels_last":  # BHWDC
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[1], W=x.shape[2], D=x.shape[3], returning_active_ex=False)
                    nc = x[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else:  # channels_first, BCHWD
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=False)
                    bhwc = x.permute(0, 2, 3, 4, 1)
                    nc = bhwc[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    x = torch.zeros_like(bhwc)
                    x[ii] = nc
                    return x.permute(0, 4, 1, 2, 3)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
                    return x
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseConvNeXtLayerNorm, self).__repr__()[
               :-1] + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'


class SparseConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, exp_r=4, do_res=False, drop_path=0.,
                 layer_scale_init_value=1e-6, sparse=True):
        super().__init__()

        self.do_res = do_res
        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                                groups=in_channels)  # depthwise conv
        self.norm = SparseConvNeXtLayerNorm(in_channels, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(in_channels,
                                 exp_r * in_channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(exp_r * in_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sparse = sparse

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)  # GELU(0) == (0), so there is no need to mask x (no need to `x *= _get_active_ex_or_ii`)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        if self.sparse:
            x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=True)
        if self.do_res:
            x = input + self.drop_path(x)
        return x

    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f', sp={self.sparse})'


class SparseEncoder(nn.Module):
    def __init__(self, encoder, input_size, sbn=False, verbose=False):
        super(SparseEncoder, self).__init__()
        self.embeddings = SparseEncoder.dense_model_to_sparse(m=encoder.embeddings, verbose=verbose, sbn=sbn)
        self.mae = encoder.mae
        
        # self.encoder = SparseEncoder.dense_model_to_sparse(m=encoder, verbose=verbose, sbn=sbn)
        self.input_size, self.downsample_raito, self.enc_feat_map_chs = input_size, encoder.get_downsample_ratio(), encoder.get_feature_map_channels()

    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        oup = m
        if isinstance(m, nn.Conv3d):
            m: nn.Conv3d
            bias = m.bias is not None
            oup = SparseConv3d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool3d):
            m: nn.MaxPool3d
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation,
                                   return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool3d):
            m: nn.AvgPool3d
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode,
                                   count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            m: nn.BatchNorm3d
            oup = (SparseSyncBatchNorm3d if sbn else SparseBatchNorm3d)(m.weight.shape[0], eps=m.eps,
                                                                        momentum=m.momentum, affine=m.affine,
                                                                        track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        elif isinstance(m, nn.InstanceNorm3d):
            m: nn.InstanceNorm3d
            oup = SparseInstanceNorm3d(m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine,
                                       track_running_stats=m.track_running_stats)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            m: nn.LayerNorm
            oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, (nn.Conv1d,)):
            m: nn.Conv1d
            bias = m.bias is not None
            oup = nn.Conv1d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode)
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup

    def forward(self, x, active_b1fff):
        x1, x2, x3, x4, x5 = self.embeddings(x)
        _x5 = self.mae(x5, active_b1fff)
        return [x1, x2, x3, x4, _x5]


if __name__ == '__main__':
    x = torch.randn([1, 96, 24, 24, 24])
    _cur_active = torch.randn([1, 1, 96 // 16, 96 // 16, 96 // 16])
    print(x.shape)
    print(_get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], D=x.shape[4], returning_active_ex=True).shape)
    print(x.shape)
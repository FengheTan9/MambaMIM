import torch
import torch.nn as nn
from models.encoder import SparseConvNeXtLayerNorm, _get_active_ex_or_ii
from typing import Optional, Sequence, Tuple, Union, List
import numpy as np
from models.mamba.bi_vision_mamba import Mamba
from monai.networks.blocks.unetr_block import UnetrUpBlock

def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=0, temperature=10000.):
    grid_size = (grid_size, grid_size, grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)],
        dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v1",
        )
        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=2 * dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.mamba(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class MaskedAutoencoderMamba(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=96, downsample_rato=16, embed_dim=384, depth=8, norm_layer=nn.LayerNorm, sparse=True):
        super().__init__()
        print("mamba sparse: ", sparse)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.grid_size = img_size // downsample_rato
        self.num_patches = (self.grid_size) ** 3
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            MambaLayer(dim=embed_dim)
            for i in range(depth)])
        # self.gsc = GSC(in_channels=embed_dim, sparse=sparse)
       
        self.sparse = sparse
        if self.sparse:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = build_3d_sincos_position_embedding(self.grid_size, self.embed_dim)
        self.pos_embed.data.copy_(pos_embed)
        if self.sparse:
            torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, enc, active_b1fff):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = enc.shape  # batch, length, dim
        mask = torch.tensor(active_b1fff, dtype=torch.int).flatten(2).transpose(1, 2)
        # sort noise for each sample
        noise = 1 - mask
        len_keep = torch.sum(mask)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(enc, dim=1, index=ids_keep.repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        return x_masked, mask, ids_restore

    def unmasking(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.repeat(1, 1, x.shape[2]))  # unshuffle
        return x

    def forward_encoder(self, enc, active_b1fff=None):
        # enc = self.gsc(enc)
        B, C, H, W, D = enc.shape
        x = enc.flatten(2).transpose(1, 2)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        if self.sparse:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, active_b1fff)
            # apply Mamba blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.unmasking(x, ids_restore)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = x.transpose(1, 2).reshape(B, C, H, W, D)
        return x

    def forward(self, imgs, active_b1fff=None):
        return self.forward_encoder(imgs, active_b1fff)


class MedNeXtBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 n_groups: int or None = None,
                 sparse=False):

        super().__init__()

        self.do_res = do_res
        self.sparse = sparse
        conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.

        self.norm = SparseConvNeXtLayerNorm(normalized_shape=in_channels, data_format='channels_first', sparse=sparse)

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.sparse:
            x1 *= _get_active_ex_or_ii(H=x1.shape[2], W=x1.shape[3], D=x1.shape[4], returning_active_ex=True)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False, sparse=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, sparse=sparse)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
            self,
            sparse: bool,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.norm1 = SparseConvNeXtLayerNorm(normalized_shape=out_channels, data_format='channels_first', sparse=sparse)
        self.norm2 = SparseConvNeXtLayerNorm(normalized_shape=out_channels, data_format='channels_first', sparse=sparse)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride)
            self.norm3 = SparseConvNeXtLayerNorm(normalized_shape=out_channels, data_format='channels_first', sparse=sparse)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3,
                 do_res=True, sparse=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, sparse=sparse)

        self.resample_do_res = do_res

        conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
        res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
        x1 = x1 + res
        return x1


class UnetOutBlock(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            n_classes,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inp):
        return self.conv(inp)


class Embeddings(nn.Module):
    def __init__(self,
                 in_channel: int = 3,
                 channels: Tuple = (32, 64, 96, 128, 192),
                 depths: Tuple = (1, 1, 3, 1, 1),
                 kernels: Tuple = (3, 3, 3, 3, 3),
                 exp_r: Tuple = (2, 4, 4, 4, 2),
                 sparse=True):
        super(Embeddings, self).__init__()
        self.dim = [channels[1], channels[2], channels[3], channels[4], channels[4]]
        self.stem = nn.Conv3d(in_channels=in_channel, out_channels=channels[0], kernel_size=3, stride=1, padding=1)


        self.layer2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=channels[1],
                out_channels=channels[1],
                exp_r=exp_r[1],
                kernel_size=kernels[1],
                do_res=True,
                sparse=sparse
            )
            for i in range(depths[1])])

        self.layer3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=channels[2],
                out_channels=channels[2],
                exp_r=exp_r[2],
                kernel_size=kernels[2],
                do_res=True,
                sparse=sparse
            )
            for i in range(depths[2])])

        self.layer4 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=channels[3],
                out_channels=channels[3],
                exp_r=exp_r[3],
                kernel_size=kernels[3],
                do_res=True,
                sparse=sparse
            )
            for i in range(depths[3])])

        self.layer5 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=channels[4],
                out_channels=channels[4],
                exp_r=exp_r[4],
                kernel_size=kernels[4],
                do_res=True,
                sparse=sparse
            )
            for i in range(depths[4])])

        self.down = nn.MaxPool3d((2, 2, 2))
        self.expend1 = nn.Conv3d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=1, padding=1)
        self.expend2 = nn.Conv3d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=1, padding=1)
        self.expend3 = nn.Conv3d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=1, padding=1)
        self.expend4 = nn.Conv3d(in_channels=channels[3], out_channels=channels[4], kernel_size=3, stride=1, padding=1)

        self.encoder1 = UnetResBlock(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            sparse=sparse
        )
        self.encoder2 = UnetResBlock(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            sparse=sparse
        )
        self.encoder3 = UnetResBlock(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            sparse=sparse
        )
        self.encoder4 = UnetResBlock(
            in_channels=channels[4],
            out_channels=channels[4],
            kernel_size=3,
            stride=1,
            sparse=sparse
        )



    def forward(self, x):
        x = self.stem(x)

        x1 = self.expend1(x)

        x = self.down(x1)
        x = self.layer2(x)
        x2 = self.expend2(x)

        x = self.down(x2)
        x = self.layer3(x)
        x3 = self.expend3(x)

        x = self.down(x3)
        x = self.layer4(x)
        x4 = self.expend4(x)

        x = self.down(x4)
        x5 = self.layer5(x)

        return self.encoder1(x1), self.encoder2(x2), self.encoder3(x3), self.encoder4(x4), x5


class Encoder(nn.Module):

    def __init__(self,
                 in_channel: int = 1,
                 channels=(32, 64, 128, 192, 384),
                 depths=(1, 2, 2, 2, 1),
                 kernels=(3, 3, 3, 3, 3),
                 exp_r=(2, 2, 4, 4, 4),
                 img_size=96,
                 depth=4,
                 norm_layer=nn.LayerNorm,
                 sparse=False):
        super(Encoder, self).__init__()
        self.dim = [channels[1], channels[2], channels[3], channels[4], channels[4]]

        self.embeddings = Embeddings(in_channel=in_channel,
                                     channels=channels,
                                     depths=depths,
                                     kernels=kernels,
                                     exp_r=exp_r,
                                     sparse=sparse)

        self.mae = MaskedAutoencoderMamba(
            img_size=img_size,
            downsample_rato=self.get_downsample_ratio(),
            embed_dim=channels[-1],
            depth=depth,
            norm_layer=norm_layer,
            sparse=sparse)

    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 16

    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return self.dim

    def forward(self, x, active_b1fff=None):
        x1, x2, x3, x4, x5 = self.embeddings(x)
        _x5 = self.mae(x5, active_b1fff)
        return x1, x2, x3, x4, _x5


class Decoder(nn.Module):
    def __init__(self,
                 n_classes: int = 3,
                 channels: Tuple = (32, 64, 128, 196, 384),
                 norm_name = "instance",
                 res_block: bool = True):
        super(Decoder, self).__init__()

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=channels[4],
            out_channels=channels[4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=channels[4],
            out_channels=channels[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=channels[3],
            out_channels=channels[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=channels[2],
            out_channels=channels[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetResBlock(
            in_channels=channels[1],
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            sparse=False
        )
        self.out = UnetOutBlock(in_channels=channels[0], n_classes=n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        d4 = self.decoder5(x5, x4)
        d3 = self.decoder4(d4, x3)
        d2 = self.decoder3(d3, x2)
        d1 = self.decoder2(d2, x1)
        d0 = self.decoder1(d1)
        return self.out(d0)


class Hybird(nn.Module):
    def __init__(self,
                 in_channel: int = 3,
                 n_classes: int = 3,
                 channels: Tuple = (32, 64, 96, 128, 192),
                 depths: Tuple = (1, 1, 3, 3, 1),
                 kernels: Tuple = (3, 3, 3, 3, 3),
                 exp_r: Tuple = (2, 4, 4, 4, 2),
                 img_size=96,
                 depth=3,
                 norm_layer=nn.LayerNorm, ):
        super().__init__()
        self.embeddings = Embeddings(in_channel=in_channel,
                                     channels=channels,
                                     depths=depths,
                                     kernels=kernels,
                                     exp_r=exp_r,
                                     sparse=False)

        self.mae = MaskedAutoencoderMamba(
            img_size=img_size,
            downsample_rato=16,
            embed_dim=channels[-1],
            depth=depth,
            norm_layer=norm_layer,
            sparse=False)

        self.decoder = Decoder(
            n_classes=n_classes,
            channels=channels,
        )

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.embeddings(x)
        x5 = self.mae(x5, None)
        return self.decoder(x1, x2, x3, x4, x5)


def build_hybird(in_channel=1, n_classes=14, img_size=96):
    return Hybird(in_channel=in_channel,
                  n_classes=n_classes,
                  channels=(32, 64, 128, 192, 384),
                  depths=(1, 2, 2, 2, 1),
                  kernels=(3, 3, 3, 3, 3),
                  exp_r=(2, 2, 4, 4, 4),
                  img_size=img_size,
                  depth=4)


if __name__ == '__main__':
    x = torch.rand((1, 1, 96, 96, 96))
    network = build_hybird()
    print(network(x).shape)



import torch
from timm.loss import SoftTargetCrossEntropy
from timm.models.layers import drop


from models.network.hymamba import Encoder




# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, drop.DropPath):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


pretrain_default_model_kwargs = {
    'mambamim': dict(sparse=True, drop_path_rate=0.1),
}
for kw in pretrain_default_model_kwargs.values():
    kw['pretrained'] = False
    kw['num_classes'] = 0
    kw['global_pool'] = ''



def build_sparse_encoder(name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False):
    from models.encoder import SparseEncoder
    kwargs = pretrain_default_model_kwargs[name]
    if drop_path_rate != 0:
        kwargs['drop_path_rate'] = drop_path_rate
    print(f'[build_sparse_encoder] model kwargs={kwargs}')
    encoder = Encoder(
                in_channel=1,
                channels=(32, 64, 128, 192, 384),
                depths=(1, 2, 2, 2, 1),
                kernels=(3, 3, 3, 3, 3),
                exp_r=(2, 2, 4, 4, 4),
                img_size=96,
                depth=4,
                sparse=True)
    return SparseEncoder(encoder=encoder, input_size=input_size, sbn=sbn, verbose=verbose)
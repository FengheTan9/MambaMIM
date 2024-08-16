import json
import os
import sys

from tap import Tap

import dist


class Args(Tap):
    # environment
    exp_name: str = 'mamba'
    exp_dir: str = ''   # will be created if not exists
    data_path: str = ''
    init_weight: str = ''   # use some checkpoint as model weight initialization; ONLY load model weights
    resume_from: str = ''   # resume the experiment from some checkpoint.pth; load model weights, optimizer states, and last epoch
    
    # MambaMIM hyperparameters
    mask: float = 0.75  # mask ratio, should be in (0, 1)
    
    # encoder hyperparameters
    model: str = 'mambamim'
    input_size: int = 96
    sbn: bool = True
    
    # data hyperparameters
    bs: int = 1
    dataloader_workers: int = 8
    
    # pre-training hyperparameters
    dp: float = 0.0
    base_lr: float = 1e-4
    wd: float = 0.04
    wde: float = 0.2
    ep: int = 100
    wp_ep: int = 40
    clip: int = 5.
    opt: str = 'adamw'
    ada: float = 0.
    
    # NO NEED TO SPECIFIED; each of these args would be updated in runtime automatically
    lr: float = 1e-4
    batch_size_per_gpu: int = 0
    glb_batch_size: int = 0
    densify_norm: str = ''
    device: str = 'gpu'
    local_rank: int = 0
    cmd: str = ' '.join(sys.argv[1:])
    commit_id: str = os.popen(f'git rev-parse HEAD').read().strip() or '[unknown]'
    commit_msg: str = (os.popen(f'git log -1').read().strip().splitlines() or ['[unknown]'])[-1].strip()
    last_loss: float = 0.
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    first_logging: bool = True
    log_txt_name: str = '{args.exp_dir}/pretrain_log.txt'
    tb_lg_dir: str = ''     # tensorboard log directory
    
    @property
    def is_convnext(self):
        return 'convnext' in self.model or 'cnx' in self.model
    
    @property
    def is_resnet(self):
        return 'resnet' in self.model
    
    def log_epoch(self):
        if not dist.is_local_master():
            return
        
        if self.first_logging:
            self.first_logging = False
            with open(self.log_txt_name, 'w') as fp:
                json.dump({
                    'name': self.exp_name, 'cmd': self.cmd, 'git_commit_id': self.commit_id, 'git_commit_msg': self.commit_msg,
                    'model': self.model,
                }, fp)
                fp.write('\n\n')
        
        with open(self.log_txt_name, 'a') as fp:
            json.dump({
                'cur_ep': self.cur_ep,
                'last_L': self.last_loss,
                'rema': self.remain_time, 'fini': self.finish_time,
            }, fp)
            fp.write('\n')


def init_dist_and_get_args():
    from utils import misc
    
    # initialize
    args = Args(explicit_bool=True).parse_args()
    e = os.path.abspath(args.exp_dir)
    d, e = os.path.dirname(e), os.path.basename(e)
    e = ''.join(ch if (ch.isalnum() or ch == '-') else '_' for ch in e)
    args.exp_dir = os.path.join(d, e)
    
    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_txt_name = os.path.join(args.exp_dir, 'pretrain_log.txt')
    args.tb_lg_dir = args.tb_lg_dir or os.path.join(args.exp_dir, 'tensorboard_log')
    try:
        os.makedirs(args.tb_lg_dir, exist_ok=True)
    except:
        pass
    
    misc.init_distributed_environ(exp_dir=args.exp_dir)
    
    # update args
    if not dist.initialized():
        args.sbn = False
    args.first_logging = True
    args.device = dist.get_device()
    args.batch_size_per_gpu = args.bs // dist.get_world_size()
    args.glb_batch_size = args.batch_size_per_gpu * dist.get_world_size()
    

    args.ada = args.ada or 0.999
    args.densify_norm = 'ln'
    
    args.opt = args.opt.lower()
    args.lr = args.base_lr
    args.wde = args.wde or args.wd
    
    return args

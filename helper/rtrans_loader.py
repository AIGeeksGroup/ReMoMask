'''
python helper/rtrans_loader.py \
--gpu_id 0
'''

import os
import sys
sys.path.insert(0, os.getcwd())
# --------------
from utils.get_opt import get_opt
from os.path import join as pjoin
from options.eval_option import EvalT2MOptions
# from eval_res import load_res_aux, load_res_ts
import torch
from models.transformer.transformer_aux import MaskTransformer, ResidualTransformer
from models.transformer.transformer_ts import MaskTransformer2D, ResidualTransformer2D


def load_res_aux(res_opt, which_model, vq_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens1d = vq_opt.nb_code1d
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim1d,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', which_model),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer_aux'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_res_ts(res_opt, which_model, vq_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens2d = vq_opt.nb_code2d
    res_transformer = ResidualTransformer2D(code_dim=vq_opt.code_dim2d,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', which_model),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer_ts'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    msg = f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!', 'Value: ', ckpt['best_value'] if 'best_value' in ckpt.keys() else ''
    print(msg)
    # print(msg, file=f, flush=True)
    return res_transformer

clip_version = 'ViT-B/32'

parser = EvalT2MOptions()
opt = parser.parse()

opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
torch.autograd.set_detect_anomaly(True)

vq_opt_path = pjoin('logs/humanml3d/pretrain_vq', 'opt.txt')
vq_opt = get_opt(vq_opt_path, device=opt.device)
# vq_model, vq_opt = load_vq_model(vq_opt)

res_opt_path = pjoin('logs/humanml3d/pretrain_rtrans', 'opt.txt')
res_opt = get_opt(res_opt_path, device=opt.device)
# res_opt.num_tokens2d = vq_model.num_code2d
res_opt.num_tokens2d = 256


which_model = 'net_best_fid.tar'
print('loading checkpoint {}'.format(which_model))
res_transformer_aux = load_res_aux(res_opt, which_model, vq_opt)
res_transformer_ts = load_res_ts(res_opt, which_model, vq_opt)

import ipdb; ipdb.set_trace()

# for file in os.listdir(model_dir):
#     if opt.which_epoch != "all" and opt.which_epoch not in file:
#         continue
#     print('loading checkpoint {}'.format(file))
#     if not opt.traverse_res:
#         mask_transformer_aux = load_trans_aux(model_opt, file)
#         mask_transformer_ts = load_trans_ts(model_opt, file)
#     else:
#         res_transformer_aux = load_res_aux(res_opt, file)
#         res_transformer_ts = load_res_ts(res_opt, file)



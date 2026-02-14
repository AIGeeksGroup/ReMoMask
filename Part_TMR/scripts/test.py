'''
python Part_TMR/scripts/test.py \
    device=cuda:4 \
    exp_name=exp_pretrain
'''

import logging
import os
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
from Part_TMR.datasets import TextMotionDataset
from Part_TMR.scripts.utils import set_seed

log = logging.getLogger(__name__)

# ********** global config ***********
from Part_TMR.models.builder_bimoco import MoCoTMR
from config import global_momentum_config
# global_device = "cuda:7" if torch.cuda.is_available() else "cpu"
# ********** global config ***********


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.train.seed)

    test_dataloader = prepare_test_dataset(cfg)
    model = prepare_test_model(cfg)
    print(f"----------- model.device: {model.device} ------------")
    eval(cfg, test_dataloader, model)
    print(f"----------- model.device: {model.device} ------------")


def prepare_test_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std.npy"))

    if cfg.eval.eval_train:
        test_split_file = pjoin(cfg.dataset.data_root, cfg.dataset.train_split_file)
    else:
        test_split_file = pjoin(cfg.dataset.data_root, "test.txt")

    test_dataset = TextMotionDataset(
        cfg,
        mean,
        std,
        test_split_file,
        eval_mode=True,
        fps=True,
    )

    # test_dataloader
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16
    )

    return test_dataloader


def prepare_test_model(cfg):
    text_encoder_alias = cfg.model.text_encoder
    text_encoder_trainable: bool = False
    motion_embedding_dims: int = cfg.model.motion_embedding_dims
    text_embedding_dims: int = cfg.model.text_embedding_dims
    projection_dims: int = cfg.model.projection_dims

    model = MoCoTMR(
        text_encoder_alias,
        text_encoder_trainable,
        motion_embedding_dims,
        text_embedding_dims,
        projection_dims,
        mode="t2m" if cfg.dataset.dataset_name == "HumanML3D" else "kit",
        config = cfg
    )

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if cfg.eval.use_best_model:
        model_path = pjoin(cfg.checkpoints_dir, "best_model.pt")
    else:
        model_path = pjoin(cfg.checkpoints_dir, "last_model.pt")

    print(model_path)
    state_dict = torch.load(model_path)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # assert all(k.startswith('generator.') for k in unexpected_keys)
    # assert len(missing_keys) == 0

    model.to(cfg.device).eval()

    return model


def eval(cfg, test_dataloader, model, verbose=True):
    device = cfg.device

    dataset_pair = dict()

    all_imgs_feat = []
    all_captions_feat = []

    all_img_idxs = []
    all_captions = []

    step = 0

    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            step += 1

            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, texts, m_length, img_indexs = (
                batch
            )

            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = (
                Root.to(device),
                R_Leg.to(device),
                L_Leg.to(device),
                Backbone.to(device),
                R_Arm.to(device),
                L_Arm.to(device),
            )

            motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

            # tokenize
            texts_token = model.tokenize(texts)
            texts_token = texts_token.to(device)

            # a. encode ---------
            motion_dict = model.encode_motion(motions)  # dict with 'global', 'parts', 'concat'
            motion_features = motion_dict['global']    # (b, c)
            text_features = model.encode_text(texts_token)   # (b, c)

            # b. morm ---------
            motion_features = motion_features / motion_features.norm(
                dim=1, keepdim=True
            )
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # c. save： ---------
            # append the features of each sample to all imgs feat, all captions feat.
            for i in range(motion_features.size(0)):
                all_imgs_feat.append(motion_features[i].cpu().numpy())
                all_captions_feat.append(text_features[i].cpu().numpy())

                all_captions.append(texts[i])
                all_img_idxs.append(img_indexs[i].item())

    all_captions = np.array(all_captions)   # (B, ) 
    for img_idx, caption in zip(all_img_idxs, all_captions):
        dataset_pair[img_idx] = np.where(all_captions == caption)[0]

    all_imgs_feat = np.vstack(all_imgs_feat)
    all_captions_feat = np.vstack(all_captions_feat)  

    sims_t2m = 100 * all_captions_feat.dot(all_imgs_feat.T)

    t2m_r1 = 0
    # Text->Motion
    ranks = np.zeros(sims_t2m.shape[0])
    for index, score in enumerate(tqdm(sims_t2m)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            t2m_r1 = r
        if verbose:
            log.info(f"t2m_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"t2m_recall_median_correct_composition: {np.median(ranks)+1:.2f}")


    # match motions queries to target texts, get nearest neighbors
    sims_m2t = sims_t2m.T

    m2t_r1 = 0
    # Motion->Text
    ranks = np.zeros(sims_m2t.shape[0])
    for index, score in enumerate(tqdm(sims_m2t)):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in dataset_pair[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    for k in [1, 2, 3, 5, 10]:
        # Compute metrics
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            m2t_r1 = r
        if verbose:
            log.info(f"m2t_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"m2t_recall_median_correct_composition: {np.median(ranks)+1:.2f}")

    return t2m_r1, m2t_r1


if __name__ == "__main__":
    main()


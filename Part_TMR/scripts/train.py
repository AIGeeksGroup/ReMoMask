
import itertools
import logging
import os
import random
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from Part_TMR.datasets import TextMotionDataset
from Part_TMR.scripts.test import eval, prepare_test_dataset
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
log = logging.getLogger(__name__)


# *********** global config  **********
from Part_TMR.models.builder_bimoco import MoCoTMR
from config import global_momentum_config
device = "cuda:0" if torch.cuda.is_available() else "cpu"  
# *********** global config  **********


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg))

    # set_seed(cfg.train.seed)
    train_dataloader, test_dataloader = prepare_dataset(cfg)  
    eval_dataloader = prepare_test_dataset(cfg)
    model, optimizer, scheduler, tokenizer = prepare_model(cfg, train_dataloader)
    train(
        cfg,
        train_dataloader,
        test_dataloader,
        eval_dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
    )


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def prepare_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std.npy"))
    
    train_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    train_dataset = TextMotionDataset(
        cfg,
        mean,
        std,
        train_split_file,
        fps=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
    )

    val_split_file = pjoin(cfg.dataset.data_root, "val.txt")
    val_dataset = TextMotionDataset(
        cfg,
        mean,
        std,
        val_split_file,
        fps=True,
    )
    test_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=16
    )

    return train_dataloader, test_dataloader


def prepare_model(cfg, train_dataloader):

    text_encoder_alias = cfg.model.text_encoder
    text_encoder_trainable: bool = cfg.train.train_text_encoder
    motion_embedding_dims: int = 512
    text_embedding_dims: int = 768
    projection_dims: int = 512   # 256 -> 512

    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_alias, TOKENIZERS_PARALLELISM=True
    )
    
    # ***** load MoCoTMR ***** 
    model = MoCoTMR(
        text_encoder_alias,
        text_encoder_trainable,
        motion_embedding_dims,
        text_embedding_dims,
        projection_dims,
        dropout=0.5 if cfg.dataset.dataset_name == "HumanML3D" else 0.0,
        mode="t2m" if cfg.dataset.dataset_name == "HumanML3D" else "kit",
        temp = 0.07,
        config = global_momentum_config
    )

    model.to(device)
    parameters = [
        {
            "params": model.motion_encoder.parameters(),
            "lr": cfg.train.optimizer.motion_lr * cfg.dataset.motion_lr_factor,
        },
        {
            "params": model.text_encoder.parameters(),
            "lr": cfg.train.optimizer.text_lr * cfg.dataset.text_lr_factor,
        },
        {
            "params": itertools.chain(
                model.motion_projection.parameters(),
                model.text_projection.parameters(),
            ),
            "lr": cfg.train.optimizer.head_lr * cfg.dataset.head_lr_factor,
        },
    ]

    optimizer = optim.Adam(parameters)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader) * cfg.train.epoch * 2
    )

    return model, optimizer, scheduler, tokenizer

def estimate_eta(start_time, current_step, total_steps):
    elapsed = time.time() - start_time
    avg_time = elapsed / (current_step + 1e-8)
    remaining_time = avg_time * (total_steps - current_step)
    return time.strftime("%H:%M:%S", time.gmtime(remaining_time))


def train(
    cfg,
    train_dataloader,
    test_dataloader,
    eval_dataloader,
    model,
    tokenizer,
    optimizer,
    scheduler,
):

    best_te_loss = 1e5
    best_t2m_r1 = 0
    best_m2t_r1 = 0
    best_h_r1 = 0
    best_ep = -1

    total_train_steps = len(train_dataloader) * cfg.train.epoch
    train_start = time.time()

    for epoch in range(cfg.train.epoch):
        print(
            f"running epoch {epoch}, best test loss {best_te_loss} best_t2m_r1 {best_t2m_r1} best_m2t_r1 {best_m2t_r1} after epoch {best_ep}"
        )
        step = 0
        tr_loss = 0
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        
        # 训练batch
        for batch in pbar:
            step += 1
            train_step_start = time.time()

            optimizer.zero_grad()

            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, texts, _, _ = batch
            # Root.shape: (128, 224, 7)  # (b, T, part_dim)

            # captions for generator
            captions = list(texts)  # list[str]  (b,)
    
            Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = (
                Root.to(device),
                R_Leg.to(device),
                L_Leg.to(device),
                Backbone.to(device),
                R_Arm.to(device),
                L_Arm.to(device),
            )
            motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

            texts = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            
            total_loss = model(motions, texts, captions, return_loss=True)

            total_loss.backward()

            tr_loss += total_loss.item()
            optimizer.step()  
            scheduler.step()  
            train_step_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_step_start ))
            eta = estimate_eta(train_start, epoch * step + step, total_train_steps)
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start))
            pbar.set_description(
                f"bacthCE: {total_loss.item():.4f} | Step: {train_step_time} | Elapsed: {elapsed_time} | ETA: {eta}"
            )


        tr_loss /= step

        # At the end of each epoch, calculate the loss of the test set.
        step = 1
        te_loss = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:

                step += 1
                Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, texts, _, _ = batch

                # captions for generator
                captions = list(texts)  # list[str]  (b,)

                Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = (
                    Root.to(device),
                    R_Leg.to(device),
                    L_Leg.to(device),
                    Backbone.to(device),
                    R_Arm.to(device),
                    L_Arm.to(device),
                )
                motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

                texts = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

                total_loss = model(motions, texts, captions, return_loss=True)

                te_loss += total_loss.item()
                test_pbar.set_description(
                    f"test batchCE: {total_loss.item()}", refresh=True
                )
            te_loss /= step

        if te_loss < best_te_loss:
            best_te_loss = te_loss

        torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "last_model.pt"))

        t2m_r1, m2t_r1 = eval(
            cfg, eval_dataloader, model, tokenizer=tokenizer, verbose=False, device_=device
        )

        log.info(
            f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}, t2m_r1 {t2m_r1}, m2t_r1 {m2t_r1} "
        )

        best_t2m_r1 = max(best_t2m_r1, t2m_r1)
        best_m2t_r1 = max(best_m2t_r1, m2t_r1)

        # if best_m2t_r1 == m2t_r1:
        #     best_ep = epoch
        #     torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "best_model.pt"))

        if best_t2m_r1 == t2m_r1 and abs(t2m_r1 - m2t_r1) < 10: 
            best_ep = epoch
            torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "best_model.pt"))


if __name__ == "__main__":
    main()
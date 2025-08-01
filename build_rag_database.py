'''
Overview of database files generated after running the script
File Path	               Type	                    Description
database/motion_ids.npy	   List[str]	                Names of all motion samples; each item is in the form of motionid_index (e.g., 001_0)
database/all_captions.npy  List[str]	                Natural language description (caption) corresponding to each motion
database/encoded_motions.npy	np.ndarray (N, 1, D)	Vector representation of each motion, obtained from model.encode_motion()
database/encoded_texts.npy	  np.ndarray (N, 1, D)	Vector representation of each caption, obtained from model.encode_text()
database/tag_lists.npy	   List[[float, float]]	    Optional: Start and end time tags for each motion
database/motion_tokens.npy	 Dict[str, np.ndarray]	Optional: VQ-VAE token sequence for each motion, used for generation
'''


import codecs as cs
import logging
import os
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
# from pytorch_lightning import seed_everything
from tqdm import tqdm

from omegaconf import OmegaConf
from transformers import AutoTokenizer


from Part_TMR.models.builder_bimoco import MoCoTMR
from Part_TMR.datasets.utils import whole2parts


logger = logging.getLogger(__name__)

tmr_model_path = "Part_TMR/checkpoints/exp1/HumanML3D"
data_root = "dataset/HumanML3D"
min_motion_length = 40  # the minimum motion length. Original motions less than 40 frames will be ignored
fps = 20    # The frame rate of the HumanML3D dataset is 20 frames per second
unit_length = 4  # 4 frames per second. After VQ-VAE encoding, each token represents 4 frames

split_file = pjoin(data_root, "train.txt")   # 'datasets/humanml3d/train.txt'
motion_dir = pjoin(data_root, "new_joint_vecs") # 'datasets/humanml3d/new_joint_vecs'
text_dir = pjoin(data_root, "texts")    # 'datasets/humanml3d/texts'
device = "cuda"

def prepare_test_model(cfg):
    '''
    加载 TMR 模型和 tokenizer
    '''

    text_encoder_alias = cfg.model.text_encoder
    text_encoder_trainable: bool = cfg.train.train_text_encoder
    motion_embedding_dims: int = 512
    text_embedding_dims: int = 768
    projection_dims: int = 512   # 256 -> 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)


    # 1. 实例化
    config = {
        'embed_dim' : 512,
        'queue_size' : 65536,  # must can be divided by RAG batch_size when training RAG model
        'momentum' : 0.99
    }

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
        alpha = 0.9996,
        config = config
    )


    if cfg.eval.use_best_model:
        model_path = pjoin(tmr_model_path, "best_model.pt")
    else:
        model_path = pjoin(tmr_model_path, "last_model.pt")

    print(model_path)
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.to(device)

    return model, tokenizer

def encode_motion():
    cfg = OmegaConf.load(pjoin(tmr_model_path, ".hydra/config.yaml"))  # 'Part_TMR/checkpoints/exp1/HumanML3D/.hydra/config.yaml'

    print("Loading the model")
    model, tokenizer = prepare_test_model(cfg)
    model.eval()

    mean = np.load(pjoin(cfg.dataset.data_root, "Mean.npy"))   # (263,)  # './dataset/HumanML3D/Mean.npy'
    std = np.load(pjoin(cfg.dataset.data_root, "Std.npy"))     # (263,)

    motion_embeddings = []
    text_embeddings = []
    caption_list = []
    motion_ids = []
    tag_lists = []

    # print(split_file)   # 'datasets/humanml3d/train.txt'
    for name in tqdm(open(split_file, "r").readlines()):
        name = name.strip()  # '000000'

        # ------------------- raw motion -------------------
        raw_motion = np.load(pjoin(motion_dir, name + ".npy"))   # (116, 263) 
        # ------------------- raw motion -------------------


        if (len(raw_motion)) < min_motion_length:  #  min_motion_length = 40
            continue
        if np.isnan(raw_motion).any():
            print(name)
            continue

        # text_dir: 'datasets/humanml3d/texts'
        with cs.open(pjoin(text_dir, name + ".txt")) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line_split = line.strip().split("#")
                # ['a man kicks something or someone with his left leg.', 'a/DET man/NOUN kick/VERB something/PRON or/CCONJ someone/PRON with/ADP his/DET left/ADJ leg/NOUN', '0.0', '0.0']
                caption = line_split[0]     # 'a man kicks something or someone with his left leg.'
                t_tokens = line_split[1].split(" ")    # ['a/DET', 'man/NOUN', 'kick/VERB', 'something/PRON', 'or/CCONJ', 'someone/PRON', 'with/ADP', 'his/DET', 'left/ADJ', 'leg/NOUN']
                f_tag = float(line_split[2])        # 0.0
                to_tag = float(line_split[3])       # 0.0
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                # debug: f_tag = 0, to_tag = 0
                if f_tag == 0.0 and to_tag == 0.0:
                    motion = raw_motion
                else:
                    # fps = 20， that is 20 frame per second
                    motion = raw_motion[int(f_tag * fps) : int(to_tag * fps)]  #

                motion = torch.from_numpy(motion).to(torch.float)
                motion = (motion - mean) / std   # (116, 263)

                m_length = motion.shape[0]      # 116

                max_motion_length = cfg.dataset.max_motion_length  # 224
                if m_length >= max_motion_length:
                    idx = 0
                    motion = motion[idx : idx + max_motion_length]
                    m_length = max_motion_length
                else:
                    padding_len = max_motion_length - m_length  # 108
                    D = motion.shape[1]     # 263
                    padding_zeros = np.zeros((padding_len, D), dtype=np.float32)  # (108, 263)
                    motion = np.concatenate((motion, padding_zeros), axis=0)    # (224, 263)


                # ------------------- Divide the motion into six parts -------------------
                Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(
                    motion,
                    mode="t2m",
                )
                '''
                Root.shape: (224, 7)
                R_Leg.shape: (224, 50)
                L_Leg.shape: (224, 50)
                Backbone.shape: (224, 60)
                R_Arm.shape: (224, 60)
                L_Arm.shape: (224, 60)
                '''       
                # ------------------- Divide the motion into six parts -------------------   
                


                Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = (
                    Root.to(device).unsqueeze(0),
                    R_Leg.to(device).unsqueeze(0),
                    L_Leg.to(device).unsqueeze(0),
                    Backbone.to(device).unsqueeze(0),
                    R_Arm.to(device).unsqueeze(0),
                    L_Arm.to(device).unsqueeze(0),
                )

                motions = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]   

                with torch.inference_mode():
                    # ---------------------------- encode motion ---------------------------------
                    motion_features = model.encode_motion(motions)  # list(tensor) -> (512 * 6) -> (bs, 512)
                    # motion_features.shape = torch.Size([1, 256])
                    # ---------------------------- encode motion ---------------------------------

                    motion_features = motion_features / motion_features.norm(
                        dim=1, keepdim=True
                    )
                    # motion_features.shape = torch.Size([1, 512])
                    motion_latent = motion_features.cpu().numpy() # (1, 512)

                with torch.inference_mode():
                    texts_token = tokenizer(
                        [caption], padding=True, truncation=True, return_tensors="pt"
                    ).to(device)   # p texts_token.input_ids.shape = torch.Size([1, 13])
                    text_features = model.encode_text(texts_token)  # torch.Size([1, 512])
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)  # torch.Size([1, 512])
                    text_latent = text_features.cpu().numpy()   # (1, 512)

                motion_embeddings.append(motion_latent)
                text_embeddings.append(text_latent)
                caption_list.append(caption)
                motion_ids.append(f"{name}_{idx}")
                tag_lists.append([f_tag, to_tag])
    
    # len(motion_embeddings) = 25
    # len(text_embeddings) = 25
    motion_embeddings = np.array(motion_embeddings)   # (25, 1, 256)
    # motion_embeddings.shape = (25, 1, 256)
    text_embeddings = np.array(text_embeddings)       # (25, 1, 256)
    # text_embeddings.shape = (25, 1, 256)

    output_folder = "database"
    os.makedirs(output_folder, exist_ok=True)

    path = os.path.join(output_folder, "all_captions.npy")
    np.save(path, caption_list)

    path = os.path.join(output_folder, "motion_ids.npy")
    np.save(path, motion_ids)

    path = os.path.join(output_folder, "tag_lists.npy")
    np.save(path, tag_lists)

    path = os.path.join(output_folder, "encoded_motions.npy")
    np.save(path, motion_embeddings)
    print(f"Encoding done, motion latent saved in:\n{path}")

    path = os.path.join(output_folder, "encoded_texts.npy")
    np.save(path, text_embeddings)
    print(f"Encoding done, text latent saved in:\n{path}")


    motion_token_dict = {}  # 存放 VQ-VAE 编码后的 token 序列

    for idx in tqdm(range(len(motion_ids))):
        motion_name = motion_ids[idx]
        f_tag, to_tag = tag_lists[idx]
        motion_token = np.load(  # 000000_0
            f"dataset/HumanML3D/TOKENS/{motion_name.split('_')[0]}.npy"  
        )[0]

        if f_tag == 0.0 and to_tag == 0.0:
            motion_token = motion_token
        else:
            # print(unit_length) :  4
            if int(f_tag * fps / unit_length) < int(to_tag * fps / unit_length):
                motion_token = motion_token[
                    int(f_tag * fps / unit_length) : int(to_tag * fps / unit_length)
                ]

        motion_token_dict[motion_name] = motion_token
    np.save(f"{output_folder}/motion_tokens.npy", motion_token_dict)  # 'database/motion_tokens.npy'


if __name__ == "__main__":
    encode_motion()

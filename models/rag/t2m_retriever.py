import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import OmegaConf
from os.path import join as pjoin
import transformers
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from typing import List

# ******** global config *********
from Part_TMR.models.builder_bimoco import MoCoTMR
from config import global_momentum_config
from config import global_bimoco_config
# **********************************

# latent_dim = 512

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        self.text_model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]


class MotionEncoder(nn.Module):
    def __init__(
        self,
        image_embedding_dim,
        mode: str = "t2m",
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.parts_name = ["Root", "R_Leg", "L_Leg", "Backbone", "R_Arm", "L_Arm"]
        if mode == "t2m":
            parts_input_dim = {
                "Root": 7,
                "R_Leg": 50,
                "L_Leg": 50,
                "Backbone": 60,
                "R_Arm": 60,
                "L_Arm": 60,
            }
        else:
            parts_input_dim = {
                "Root": 7,
                "R_Leg": 62,
                "L_Leg": 62,
                "Backbone": 48,
                "R_Arm": 48,
                "L_Arm": 48,
            }

        for name in self.parts_name:
            raw_dim = parts_input_dim[name]  # 7

            latent_dim = image_embedding_dim

            skel_embedding = nn.Linear(raw_dim, image_embedding_dim) # d'

            emb_token = nn.Parameter(torch.randn(latent_dim))

            sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

            seq_trans_encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
            )

            seqTransEncoder = nn.TransformerEncoder(
                seq_trans_encoder_layer, num_layers=num_layers
            )

            setattr(self, f"skel_embedding_{name}", skel_embedding)
            setattr(self, f"emb_token_{name}", emb_token)
            setattr(self, f"sequence_pos_encoding_{name}", sequence_pos_encoding)
            setattr(self, f"seqTransEncoder_{name}", seqTransEncoder)

        self.target_token_idx = 0

    def extarct_feature(self, x, name):

        bs, nframes, nfeats = x.shape

        skel_embedding = getattr(self, f"skel_embedding_{name}")
        # Embed each human poses into latent vectors
        x = skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        emb_token = getattr(self, f"emb_token_{name}")

        emb_token = torch.tile(emb_token, (bs,)).reshape(bs, -1)

        # adding the embedding token for all sequences
        xseq = torch.cat((emb_token[None], x), 0)

        sequence_pos_encoding = getattr(self, f"sequence_pos_encoding_{name}")
        seqTransEncoder = getattr(self, f"seqTransEncoder_{name}")

        # add positional encoding
        xseq = sequence_pos_encoding(xseq)
        final = seqTransEncoder(xseq)

        return final[self.target_token_idx]

    def forward(self, parts):
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        embedding_parts = []
        for i, name in enumerate(self.parts_name):
            embedding_parts.append(self.extarct_feature(parts[i], name))

        return torch.concatenate(embedding_parts, dim=1)



class Retriever(ABC):
    @abstractmethod
    def retrieve(self, caption: str, *args, **kwargs) -> List[int]:
        """
        retrieve the motion ids which are similar to the current caption and motion length.
        
        :param caption: prompt
        :param length: motion length
        :return: reteieved motion ids
        """
        pass


class MocoTmrRetriever(Retriever, nn.Module):
    def __init__(
        self,
        motion_codebook_size=512,
        database_path="database",   # RAG database dir 
        tmr_model_path="Part_TMR/checkpoints/exp1/HumanML3D",    # RAG model dir
        device = None,   # device
    ):
        super().__init__()
        device =  device if device else "cuda" if torch.cuda.is_available() else "cpu"

        self.m_codebook_size = motion_codebook_size
        self.num_retrieval = 10
        self.top_k = 1
        self.use_shuffle = False

        # Load retrieval database
        print("Loading motion retrieval database from", database_path)
        self.motion_names = np.load(f"{database_path}/motion_ids.npy")   # 'database/motion_ids.npy'  # (NC,)
        self.captions = np.load(f"{database_path}/all_captions.npy")     # (NC,)
        motion_features = np.load(f"{database_path}/encoded_motions.npy")[:, 0, :]  # (NC, latent_dim)
        # print(motion_features.shape)  # (25, 512)
        text_features = np.load(f"{database_path}/encoded_texts.npy")[:, 0, :]      # (NC, clip_dim)
        self.motion_features = torch.Tensor(motion_features)    # (NC, latent_dim)
        self.text_features = torch.Tensor(text_features)        # (NC, clip_dim)


        assert (
            self.motion_features.size(0)
            == self.text_features.size(0)
            == len(self.motion_names)
            == len(self.captions)
        )

        # Load RAG model
        cfg = OmegaConf.load(pjoin(tmr_model_path, ".hydra/config.yaml"))  # Part_TMR/checkpoints/exp1/HumanML3D/.hydra/config.yaml
        self.tmr_cfg = cfg
        text_encoder_alias = "distilbert-base-uncased" 

        # you may need to set up a proxy to download the tokenizer model 
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias, cache_dir="local_tokenizer")

        ### *************** - load RAG model - ******************
        text_encoder_alias = cfg.model.text_encoder
        text_encoder_trainable = False
        motion_embedding_dims: int = global_bimoco_config['motion_embedding_dims']
        text_embedding_dims: int = global_bimoco_config['text_embedding_dims']
        projection_dims: int = global_bimoco_config['projection_dims']

        model = MoCoTMR(
            text_encoder_alias,
            text_encoder_trainable,
            motion_embedding_dims,
            text_embedding_dims,
            projection_dims,
            dropout=0.5 if cfg.dataset.dataset_name == "HumanML3D" else 0.0,
            mode="t2m" if cfg.dataset.dataset_name == "HumanML3D" else "kit",
            config = global_momentum_config
        )

        # 2. load state_dict
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

        if cfg.eval.use_best_model:
            model_path = pjoin(cfg.checkpoints_dir, "best_model.pt")
        else:
            model_path = pjoin(cfg.checkpoints_dir, "last_model.pt")

        print(model_path)
        state_dict = torch.load(model_path, map_location=device)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert all(k.startswith('generator.') for k in unexpected_keys)
        assert len(missing_keys) == 0

        # 3. set to eval
        model.to(device)
        model.eval()

        # post process
        self.tmr_model = model
        self.tmr_tokenizer = tokenizer
        self.device = device
        self.to(self.device)

        # clean GPU memory
        torch.cuda.empty_cache()



    def cal_text_motion_sim(self, text_feature, motion_feature):
        text_feature = torch.nn.functional.normalize(text_feature, dim=-1).to(self.device)  # (1, 512)
        motion_feature = torch.nn.functional.normalize(motion_feature, dim=-1).to(self.device)  # (N, 512)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(text_feature, motion_feature)  # (N,)

    def get_knn_samples(self, caption, k=2, nn_names=None):
        '''
        caption:str
        '''
        with torch.inference_mode():
            texts_token = self.tmr_tokenizer(
                [caption], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)     # texts_token['input_ids'].shape  # (b, token_len)
            text_feature = self.tmr_model.encode_text(texts_token)[0].unsqueeze(0)   # shape: (1, 512)

        score = self.cal_text_motion_sim(text_feature, self.motion_features)  # (N,)
        indexes = torch.argsort(score, descending=True).cpu().numpy()  # (N,)

        nn_indexes = []
        # nn names maintains the list of selected motion ids, avoiding the repeated selection of multiple variants of the same motion and effectively improving diversity
        nn_names = [] if nn_names is None else nn_names
        for index in indexes:
            motion_name_ = self.motion_names[index].split("_")[0]
            if motion_name_ in nn_names:
                continue
            nn_indexes.append(index)
            nn_names.append(motion_name_)
            if len(nn_indexes) >= self.num_retrieval:
                break

        if self.use_shuffle:
            nn_indexes, nn_names = self.shuffle_samples(nn_indexes, nn_names)

        return nn_indexes[:k]

    def retrieve(self, caption: str, *args, **kwargs) -> List[int]:
        '''
        input:
            caption: str, prompt
            k: int, optional, defaultsi self.top_k
        输出:
            nn_indexes: List[int], retrieved motion indexes
        '''
        k = kwargs.get("k", self.top_k)
        return self.get_knn_samples(caption, k=k)

    def forward(self, captions: List[str], *args, **kwargs) -> dict:
        '''
        输入:
            captions: list[str]
            k: int, optional, defaultsi self.top_k
        输出:
            re_dict: dict, includes retrived text and motion features
                - re_text: Tensor, shape (b, k, 1, latent_dim)
                - re_motion: Tensor, shape (b, k, 1, latent_dim)
        '''

        # import ipdb; ipdb.set_trace()  # Debugging breakpoint
        k = kwargs.get("k", self.top_k)
        b = len(captions)  # batch_size
        all_indexes = []

        # RAG -----------
        for b_ix in range(b):
            batch_indexes = self.retrieve(captions[b_ix], k = k)
            
            # retrieved motion indexes
            all_indexes.append(batch_indexes) 
        # --- construct re_dict ---
        all_indexes = torch.tensor(all_indexes, dtype=torch.long)  # shape: (b, k)
        b, k = all_indexes.shape

        # Flatten it into a one-dimensional index and extract the corresponding items in motion features
        flat_indexes = all_indexes.view(-1)  # shape: (b * k)

        selected_motions = self.motion_features[flat_indexes].to(self.device)  # shape: (b * k, latent_dim)
        selected_texts = self.text_features[flat_indexes].to(self.device)     # shape: (b * k, latent_dim)

        # reshape back to (b, k, 1, latent_dim)
        all_motions_feature = selected_motions.view(b, k, 1, -1)  # (b, k, 1, latent_dim)
        all_texts_feature = selected_texts.view(b, k, 1, -1)      # (b, k, 1, latent_dim)

        re_dict = {
            "re_text": all_texts_feature.to(self.device),
            "re_motion": all_motions_feature.to(self.device),
        }
        return re_dict



if __name__ == "__main__":
    print("Instantiating retriever...")
    # retriever = T2MRetriever()
    # retriever = MocoTmrRetriever()
    retriever = MocoTmrRetriever(device = "cuda")   
    # retriever = MocoTmrRetriever(device = torch.device(type='cuda', index=0))   
    # retriever = MocoTmrRetriever(device = "cuda:2")   
    print("Retriever instantiated.")


    # ------------- one caption ----------------
    # test_caption = "a person walks forward and then turns around"

    # # prepare for semantic attention
    # # import ipdb; ipdb.set_trace()  
    # retrieved_idx = retriever.retrieve(test_caption, k=2)
    # retrieved_motion_feats = retriever.motion_features[retrieved_idx].to(device)  # (2, 256)  # (k, 256)
    # retrieved_text_feats = retriever.text_features[retrieved_idx].to(device)      # (2, 256)  # (k, 256)

    # print(len(retrieved_idx))           # 2
    # print(retrieved_motion_feats.shape) # (2, 512)  # R^m
    # print(retrieved_text_feats.shape)   # (2, 512)  # R^t

    # ------------- batch captions ----------------
    captions = [
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
        "a person runs quickly",
                "a person walks forward and then turns around",
        "a person jumps up and down",
    ]
    re_dict = retriever(captions, k=2)
    for k, v in re_dict.items():
        print(f"{k}: {v.shape}")

    # log -----
    # re_text: torch.Size([33, 2, 1, 512])  # (b, k, 1, d)
    # re_motion: torch.Size([33, 2, 1, 512]) # (b, k, 1, d)
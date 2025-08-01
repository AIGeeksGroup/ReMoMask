
import numpy as np
# import timm
import torch
import transformers
from torch import nn

from .positional_encoding import PositionalEncoding

import ipdb


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)  # (embedding_dim -> projection_dim)
        x = self.gelu(projected)    
        x = self.fc(x)      # (projection_dim -> projection_dim)
        x = self.dropout(x)
        x += projected      
        return self.layer_norm(x)  # (projection_dim)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        # model_name = 'distilbert-base-uncased'
        self.text_model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        # [cls] index
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)  

        last_hidden_state = output.last_hidden_state   # torch.Size([1, 13, 768])  # (bs, seq_len, hidden_size)

        # get [cls] token as text embedding
        return last_hidden_state[:, self.target_token_idx, :]   # torch.Size([1, 768])  # (bs, hidden_size)


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

        # Each part has its own independent encoder set and does not share parameters with each other.
        for name in self.parts_name:
            raw_dim = parts_input_dim[name]   # 7 or 50 or 50 or 60 or 60 or 60
            latent_dim = image_embedding_dim  # 512

            skel_embedding = nn.Linear(raw_dim, image_embedding_dim) 

            emb_token = nn.Parameter(torch.randn(latent_dim))  # (latent_dim,)

            sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

            seq_trans_encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,         # 512
                nhead=num_heads,            # 4
                dim_feedforward=ff_size,    # 1024
                dropout=dropout,        # 0.1
                activation=activation,  # gelu
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
        '''
        ###  Note: This encode method is independent of the length of the motion. No matter how long it is, cls is finally taken as the feature vector
        input:
        - x: (b, t, part_motion_dim)
        output:
        - (b, latent_dim)
        '''
        bs, nframes, nfeats = x.shape   # torch.Size([1, 224, 7])

        skel_embedding = getattr(self, f"skel_embedding_{name}")
        # Embed each human poses into latent vectors
        x = skel_embedding(x)   # (1, 224, 7) -> (1, 224, 512)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # [nframes, bs, latent_dim]

        emb_token = getattr(self, f"emb_token_{name}")  

        emb_token = torch.tile(emb_token, (bs,)).reshape(bs, -1)  # (latent_dim) -> (bs, latent_dim)  # torch.Size([1, 512])


        # adding the embedding token for all sequences
        xseq = torch.cat((emb_token[None], x), 0)  # torch.Size([225, 1, 512])

        sequence_pos_encoding = getattr(self, f"sequence_pos_encoding_{name}")
        seqTransEncoder = getattr(self, f"seqTransEncoder_{name}")

        # add positional encoding
        xseq = sequence_pos_encoding(xseq)  # xseq.shape: torch.Size([225, 1, 512])
        final = seqTransEncoder(xseq)       # final.shape: torch.Size([225, 1, 512])

        return final[self.target_token_idx] # torch.Size([1, 512])

    def forward(self, parts):
        assert isinstance(parts, list)  # p len(parts) = 6
        assert len(parts) == len(self.parts_name)

        embedding_parts = []
        for i, name in enumerate(self.parts_name):
            # self.extarct_feature(parts[i], name).shape:  torch.Size([1, 512])
            embedding_parts.append(self.extarct_feature(parts[i], name))

        return torch.concatenate(embedding_parts, dim=1)   # torch.Size([1, 3072])



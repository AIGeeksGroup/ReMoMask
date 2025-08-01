# B: batch_size
# K: negtive_queue_size
# D: raw_motion  = 263
# T: raw_motion_len 
# C: letent_dim  = 512

# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (C x K)
# m: momentum
# t: temperature

import torch
import torch.nn.functional as F
import torch.nn as nn
from Part_TMR.models.encdoc import ProjectionHead, TextEncoder, MotionEncoder
from Part_TMR.datasets.utils import whole2parts


class MoCoTMR(nn.Module):
    def __init__(self, 
        text_encoder_alias="distilbert-base-uncased",
        text_encoder_trainable: bool = True,
        motion_embedding_dims: int = 512,
        text_embedding_dims: int = 768,
        projection_dims: int = 512,
        dropout: float = 0.5,
        mode: str = "t2m",
        temp = 0.07, 
        config = None
    ) -> None:
        super().__init__()

        embed_dim = config['embed_dim']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        self.temp = temp

        # text and motion encoders 
        motion_encoder = MotionEncoder(
            image_embedding_dim=motion_embedding_dims,   # 512
            num_layers=4,
            num_heads=4,
            mode=mode,
        )

        text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder

        self.motion_projection = ProjectionHead(
            embedding_dim=motion_embedding_dims * len(motion_encoder.parts_name),
            projection_dim=projection_dims,  # 512
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,   # 768
            projection_dim=projection_dims,     # 512
            dropout=dropout,
        )

        # momentum motion encoders
        self.motion_encoder_m = MotionEncoder(
            image_embedding_dim=motion_embedding_dims,   # 512
            num_layers=4,
            num_heads=4,
            mode=mode,
        )

        self.motion_projection_m = ProjectionHead(
            embedding_dim=motion_embedding_dims * len(motion_encoder.parts_name),
            projection_dim=projection_dims,  # 512
            dropout=dropout,
        )

        # momentum text encoders
        self.text_encoder_m = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )

        self.text_projection_m = ProjectionHead(
            embedding_dim=text_embedding_dims,   # 768
            projection_dim=projection_dims,     # 512
            dropout=dropout,
        )


        # make momentum pairs
        self.model_pairs = [
                    [self.motion_encoder,self.motion_encoder_m],
                    [self.motion_projection,self.motion_projection_m],
                    [self.text_encoder,self.text_encoder_m],
                    [self.text_projection,self.text_projection_m],
        ]

        # init momentum param
        self.copy_params()

        # create the queue                           
        self.register_buffer("motion_queue", torch.randn(embed_dim, self.queue_size))  # (C, K)
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                            
        self.motion_queue = nn.functional.normalize(self.motion_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def encode_motion(self, motion):
        '''
        input:
            - motion: list[part_motion]
                - part_motion: (b, t, part_motion_dim)
        output:
            - motion_embeddings: (b, 256)
        '''
        # motion:list
        motion_features = self.motion_encoder(motion) # torch.Size([1, 3072])  # 3072 = 512 * 6
        motion_embeddings = self.motion_projection(motion_features)   # torch.Size([1, 256])
        return motion_embeddings  # (bs, 256)

    def encode_text(self, text):
        '''
        input:
            - text:dict: dict(input_ids, attention_mask)  # texts.indices: (B, ntokens)
        output:
            - (b, ntokens) -> (b, 512)
        '''
        # text["input_ids"].shape:  torch.Size([128, 40])
        text_features = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )  # torch.Size([128, 768])

        text_embeddings = self.text_projection(text_features)  # (128, 768) -> (128, 512)

        return text_embeddings  # (b, 512)

    def forward(self, motions, texts:dict, captions:list[str], return_loss=True):
        '''
        input:
            - motions: list[part_motion]   
            - texts: dict(input_ids, attention_mask)  # texts.indices: (B, ntokens)
            - captions: list[str], (B,)
        output:
            - loss
        '''

        
        B = texts['input_ids'].shape[0]
        K = self.motion_queue.shape[1]

        # encode texts --------------
        # print(f"batch_size: {len(captions)}")
        # text["input_ids"].shape : (128, ntokens)
        text_features = self.text_encoder(
            input_ids=texts["input_ids"], attention_mask=texts["attention_mask"]
        )  # torch.Size([128, 768])
        text_feat = self.text_projection(text_features)  # (128, 768) -> (128, 512)  # (B, C)

        # encode motions ----------------
        # motion:list
        motion_embeds_m = self.motion_encoder(motions) # list -> torch.Size([1, 3072])  # 3072 = 512 * 6
        motion_feat = self.motion_projection(motion_embeds_m)   # torch.Size([128, 512])  # (B, C)

        if not return_loss:
            # k: motion_features # (B, C)
            # q: text_features # (B, C)
            return motion_feat, text_feat

        # get momentum features  --------------------------------
        with torch.no_grad():
            self._momentum_update()  # momentum update

            # ------------- moco i2m ----------------
            # encode momentum motions  
            motion_embeds_m = self.motion_encoder_m(motions) # torch.Size([1, 3072])  # 3072 = 512 * 6
            motion_feat_m = self.motion_projection_m(motion_embeds_m)   # (B, C)
            motion_feat_all = torch.cat([motion_feat_m.T, self.motion_queue.clone().detach()], dim = 1)  # (C, B + K)

            # ------------- moco m2i ----------------
            # encode momentum texts
            text_embeds_m = self.text_encoder_m(
                input_ids=texts["input_ids"], attention_mask=texts["attention_mask"]
                )  # torch.Size([128, 768])
            text_feat_m = self.text_projection_m(text_embeds_m)  # (128, 768) -> (128, 512)  3# (B, C)
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1) # (C, B + K)

            # ---------- hard label -----------
            sim_hard_targets = torch.zeros(B, B + K).to(motion_embeds_m.device)   # (B, B + K)
            sim_hard_targets.fill_diagonal_(1)  # (B, B + K)
            sim_targets = sim_hard_targets



        # get MoCo loss
        sim_m2t = motion_feat @ text_feat_all / self.temp # (B, B + K)
        sim_t2m = text_feat @ motion_feat_all / self.temp # (B, B + K)
        
        
        # corss entropy loss
        loss_m2t = -torch.sum(F.log_softmax(sim_m2t, dim=1) * sim_targets, dim=1).mean()  
        loss_t2m = -torch.sum(F.log_softmax(sim_t2m, dim=1) * sim_targets, dim=1).mean()  
        # (B, B + K) dot* (B, B + K)
        
        # mean loss
        loss = (loss_m2t+loss_t2m)/2

        if self.training:
            self._dequeue_and_enqueue(motion_feat_m, text_feat_m)
        return loss
        

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

    @torch.no_grad()
    def _dequeue_and_enqueue(self, motion_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(motion_feat)  # (B*gpus, C)
        text_feats = concat_all_gather(text_feat)   # (B*gpus, C)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.motion_queue[:, ptr:ptr + batch_size] = image_feats.T  # (C, k)
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T    # (C, k)
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        

    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # 单卡
    if not torch.distributed.is_initialized():
        return tensor

    # 多卡
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0) # (B, C) -> (B*gpus, C)
    return output


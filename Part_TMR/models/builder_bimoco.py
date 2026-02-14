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
        text_encoder_alias="ViT-B-32.pt",
        text_encoder_trainable: bool = False,
        motion_embedding_dims: int = 512,
        text_embedding_dims: int = 512,
        projection_dims: int = 512,
        dropout: float = 0.5,
        mode: str = "t2m",
        temp = 0.07, 
        alpha = 0.9996,
        config = None
    ) -> None:
        super().__init__()

        # embed_dim = config['embed_dim']
        # self.queue_size = config['queue_size']
        # self.momentum = config['momentum']

        embed_dim = config.model.embed_dim
        self.queue_size = config.model.queue_size
        self.momentum = config.model.momentum

        self.temp = temp
        
        # HBM-specific config
        self.use_hbm_loss = config.get('use_hbm_loss', True)
        self.lambda_part = config.get('lambda_part', 1.0)

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

        # Legacy motion projection (for concat features)
        self.motion_projection = ProjectionHead(
            embedding_dim=motion_embedding_dims * len(motion_encoder.parts_name),
            projection_dim=projection_dims,  # 512
            dropout=dropout,
        )
        
        # # NEW: Global motion projection
        # self.motion_projection_global = ProjectionHead(
        #     embedding_dim=motion_embedding_dims,  # 512 from average pooling
        #     projection_dim=projection_dims,
        #     dropout=dropout,
        # )
        
        # # NEW: Part projection (shared across all parts)
        # self.part_projection = ProjectionHead(
        #     embedding_dim=motion_embedding_dims,  # 512
        #     projection_dim=projection_dims,
        #     dropout=dropout,
        # )
        
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

        # Legacy momentum projection
        self.motion_projection_m = ProjectionHead(
            embedding_dim=motion_embedding_dims * len(motion_encoder.parts_name),
            projection_dim=projection_dims,  # 512
            dropout=dropout,
        )
        
        # # NEW: Momentum global motion projection
        # self.motion_projection_global_m = ProjectionHead(
        #     embedding_dim=motion_embedding_dims,
        #     projection_dim=projection_dims,
        #     dropout=dropout,
        # )
        
        # # NEW: Momentum part projection
        # self.part_projection_m = ProjectionHead(
        #     embedding_dim=motion_embedding_dims,
        #     projection_dim=projection_dims,
        #     dropout=dropout,
        # )

        # momentum text encoders
        self.text_encoder_m = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )

        self.text_projection_m = ProjectionHead(
            embedding_dim=text_embedding_dims,   # 512
            projection_dim=projection_dims,     # 512
            dropout=dropout,
        )


        # make momentum pairs
        self.model_pairs = [
                    [self.motion_encoder,self.motion_encoder_m],
                    [self.motion_projection,self.motion_projection_m],
                    [self.text_encoder,self.text_encoder_m],
                    [self.text_projection,self.text_projection_m],
                    # NEW: Add global and part projection pairs
                    # [self.motion_projection_global, self.motion_projection_global_m],
                    # [self.part_projection, self.part_projection_m],
        ]

        # init momentum param
        self.copy_params()

        # create the queue
        self.register_buffer("motion_queue", torch.randn(embed_dim, self.queue_size))  # (C, K)
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
                            
        self.motion_queue = nn.functional.normalize(self.motion_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        # NEW: Part-level queues (6 queues for 6 body parts)
        num_parts = 6
        # part_queue_size = config.get('part_queue_size', self.queue_size)
        part_queue_size = self.queue_size
        self.register_buffer("part_queues", torch.randn(num_parts, embed_dim, part_queue_size))  # (6, C, K)
        self.part_queues = nn.functional.normalize(self.part_queues, dim=1) # (6, C, K)
        
        # NEW: HBM loss module
        if self.use_hbm_loss:
            from .hbm_loss import HBMLoss
            self.hbm_loss = HBMLoss(
                temperature=temp,
                lambda_part=self.lambda_part,
                num_parts=num_parts
            )

    @property
    def device(self):
        return self.text_encoder.device

    # def encode_motion(self, motion):
    #     '''
    #     input:
    #         - motion: list[part_motion]
    #             - part_motion: (b, t, part_motion_dim)
    #     output:
    #         - motion_embeddings: (b, 256)
    #     '''
    #     # motion:list
    #     motion_features = self.motion_encoder(motion) # torch.Size([1, 3072])  # 3072 = 512 * 6
    #     motion_embeddings = self.motion_projection(motion_features)   # torch.Size([1, 256])
    #     return motion_embeddings  # (bs, 256)

    def encode_motion(self, motion):
        '''
        input:
            - motion: list[part_motion]
        output:
            - motion_output: dict with 'global', 'parts', 'concat'
        '''
        motion_output = self.motion_encoder(motion)  # dict with 'global', 'parts', 'concat'
        motion_global = self.motion_projection(motion_output['concat'])  # (B, 512)
        motion_output['global'] = motion_global
        if self.use_hbm_loss:
            # HBM motion dict
            return motion_output
        else:
            # Legacy path
            motion_embeddings = self.motion_projection(motion_output['concat'])  # (B, 512)
        return motion_embeddings

    def tokenize(self, text):
        '''
        text:str or list[str]
        '''
        text_ids = self.text_encoder.tokenize(text)
        return text_ids

    def encode_text(self, text):
        '''
        input:
            - text: token_ids   #(B, ntokens)
        output:
            - (b, ntokens) -> (b, 512)
        '''
        # # text["input_ids"].shape:  torch.Size([128, 40])
        # text_features = self.text_encoder(
        #     input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        # )  # torch.Size([128, 768])

        text_features = self.text_encoder(text) 

        text_embeddings = self.text_projection(text_features)  # (128, 512) -> (128, 512)

        return text_embeddings  # (b, 512)

    def forward(self, motions, texts:dict, captions:list[str], return_loss=True):
        '''
        input:
            - motions: list[pat_motion]
            - texts: tensor of texts_ids, (B, 77)
            - captions: list[str], (B,)
        output:
            - loss (or loss_dict if using HBM loss)
        '''
        B = texts.shape[0]

        # encode texts --------------
        # text_features = self.text_encoder(
        #     input_ids=texts["input_ids"], attention_mask=texts["attention_mask"]
        # )  # torch.Size([B, 768])
        text_features = self.text_encoder(texts) # (B, 512)
        text_feat = self.text_projection(text_features)  # (B, 512) -> (B, 512)

        # encode motions ----------------
        # NEW: motion_output is now a dict
        # list -> dict
        motion_output = self.motion_encoder(motions)  # dict with 'global', 'parts', 'concat'
        
        if self.use_hbm_loss:
            # Use HBM loss with global and part features
            # motion_feat_global = motion_output['global']  # (B, 512)
            motion_feat_global = self.motion_projection(motion_output['concat'])  # (B, 512)
            
            # Project parts: (B, 6, 512) -> (B, 6, 512)
            B_check, K, D = motion_output['parts'].shape
            parts_proj = motion_output['parts'].reshape(B_check * K, D)  # (B*6, 512)
            # parts_proj = self.part_projection(parts_flat)  # (B*6, 512)
            motion_feat_parts = parts_proj.reshape(B_check, K, D)  # (B, 6, 512)
        else:
            # Legacy path: use concatenated features
            motion_feat = self.motion_projection(motion_output['concat'])  # (B, 512)

        if not return_loss:
            if self.use_hbm_loss:
                return motion_feat_global, text_feat, motion_feat_parts
            else:
                return motion_feat, text_feat

        # get momentum features  --------------------------------
        with torch.no_grad():
            self._momentum_update()  # momentum update

            # Encode momentum text
            # text_embeds_m = self.text_encoder_m(
            #     input_ids=texts["input_ids"], attention_mask=texts["attention_mask"]
            # )  # (B, 768)
            text_features = self.text_encoder_m(texts) # (B, 512)
            text_feat_m = self.text_projection_m(text_features)  # (B, 512)
            
            # Encode momentum motion
            motion_output_m = self.motion_encoder_m(motions)
            
        if self.use_hbm_loss:
            with torch.no_grad():
                # HBM loss path
                # motion_feat_global_m = motion_output_m['global']  # (B, 512)
                motion_feat_global_m = self.motion_projection_m(motion_output_m['concat'])  # (B, 512)
                
                # Project momentum parts
                parts_proj_m = motion_output_m['parts'].reshape(B * K, D)  # (B*6, 512)
                # parts_proj_m = self.part_projection_m(parts_flat_m)  # (B*6, 512)
                motion_feat_parts_m = parts_proj_m.reshape(B, K, D)  # (B, 6, 512)
                
            # Compute HBM loss
            loss_dict = self.hbm_loss(
                text_feat=text_feat,
                motion_feat_global=motion_feat_global,
                motion_feat_parts=motion_feat_parts,
                text_feat_m=text_feat_m,
                motion_feat_global_m=motion_feat_global_m,
                motion_feat_parts_m=motion_feat_parts_m,
                text_queue=self.text_queue.clone().detach(),
                motion_queue=self.motion_queue.clone().detach(),
                part_queues=self.part_queues.clone().detach(),
            )
            
            loss = loss_dict['total']
            
            # Update queues
            if self.training:
                self._dequeue_and_enqueue_hbm(motion_feat_global_m, text_feat_m, motion_feat_parts_m)
            
            return loss, loss_dict  # Return both for logging
        else:
            with torch.no_grad():
                # Legacy loss path
                motion_feat_m = self.motion_projection_m(motion_output_m['concat'])  # (B, 512)
                motion_feat_all = torch.cat([motion_feat_m.T, self.motion_queue.clone().detach()], dim=1)  # (C, B + K)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)  # (C, B + K)

                # Hard label
                K_queue = self.motion_queue.shape[1]
                sim_hard_targets = torch.zeros(B, B + K_queue).to(motion_output_m['concat'].device)
                sim_hard_targets.fill_diagonal_(1)
                sim_targets = sim_hard_targets

            # MoCo loss
            sim_m2t = motion_feat @ text_feat_all / self.temp  # (B, B + K)
            sim_t2m = text_feat @ motion_feat_all / self.temp  # (B, B + K)

            # Cross entropy loss
            loss_m2t = -torch.sum(F.log_softmax(sim_m2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2m = -torch.sum(F.log_softmax(sim_t2m, dim=1) * sim_targets, dim=1).mean()

            loss = (loss_m2t + loss_t2m) / 2

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
    def _dequeue_and_enqueue_hbm(self, motion_feat_global, text_feat, motion_feat_parts):
        """
        Update queues with global and part-level features for HBM loss.
        
        Args:
            motion_feat_global: (B, D) - global motion features
            text_feat: (B, D) - text features
            motion_feat_parts: (B, K, D) - part features where K=6
        """
        # Gather features across GPUs
        motion_feats = concat_all_gather(motion_feat_global)  # (B*GPUs, D)
        text_feats = concat_all_gather(text_feat)  # (B*GPUs, D)
        
        # Gather parts: (B, K, D) -> (B*GPUs, K, D)
        B, K, D = motion_feat_parts.shape
        parts_feats = concat_all_gather(motion_feat_parts.reshape(B, K * D))  # (B*GPUs, K*D)
        parts_feats = parts_feats.reshape(-1, K, D)  # (B*GPUs, K, D)
        
        batch_size = motion_feats.shape[0]  # B = B*GPUs
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        
        # Update global queues
        self.motion_queue[:, ptr:ptr + batch_size] = motion_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        
        # Update part queues
        part_queue_size = self.part_queues.shape[2]
        ptr_part = ptr % part_queue_size  # Handle different queue sizes
        for k in range(K):
            # self.part_queues: (6, D, K)
            # self.part_queues[k, :, ptr_part:ptr_part + batch_size]:  (D, B)
            self.part_queues[k, :, ptr_part:ptr_part + batch_size] = parts_feats[:, k, :].T  # (D, B)
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
        

    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # single gpu
    if not torch.distributed.is_initialized():
        return tensor

    # multi gpus
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0) # (B, C) -> (B*gpus, C)
    return output


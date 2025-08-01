import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# @ATTENTIONS.register_module()
class SemanticsModulatedAttention(nn.Module):

    def __init__(self, latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)
 
        self.retr_norm1 = nn.LayerNorm(2 * latent_dim)
        self.retr_norm2 = nn.LayerNorm(latent_dim)
        self.key_retr = nn.Linear(2 * latent_dim, latent_dim)
        self.value_retr = zero_module(nn.Linear(latent_dim, latent_dim))
    
    def forward(self, x, xf, src_mask, cond_type, re_dict=None):
        """
        x                       z                           (b, n, latent_dim)
        xf                      t                           (b, 1, latent_dim)
        src_mask                Valid position mask for current motion      (b, n, 1)
                                True indicates valid positions, False indicates invalid ones.
        cond_type               Integer code controlling whether to use text/retr conditions 
                                (e.g., 10 means using only retr)            # (b,) or (b, 1, 1)
        re_dict['re_motion']    Rm                          (b, k, 1, latent_dim)
        re_dict['re_text']      Rt                          (b, k, 1, latent_dim)

        return:
            y: b, n, latent_dim
        """



        xf = xf.to(x.dtype)  
        src_mask = src_mask.float()     
        
        B, T ,D = x.shape       

        re_motion = re_dict['re_motion']        
        re_text = re_dict['re_text']            
        
        N = xf.shape[1] + x.shape[1] + re_motion.shape[1] * re_motion.shape[2]   
        H = self.num_heads  


        '''
        condition control
        cond_type: (b,) or (b,1,1)
        for each batch:
            -%10 > 0 → enable retrieved text feature
            -//10 > 0 → enable retrieved motion feature
        '''        
        text_cond_type = (cond_type % 10 > 0).float()     
        retr_cond_type = (cond_type // 10 > 0).float()    
        
        '''------------ <QUERY> -----------'''
        query = self.query(self.norm(x)) * src_mask   # (b, n, latent_dim)  # z
        

        '''-------------- <KEY> ------------'''
        re_feat_key = torch.cat((re_motion, re_text), dim=-1)    
        re_feat_key = re_feat_key.reshape(B, -1, 2 * D)       # (b, k*n, latent_dim + latent_dim)  # R_m + R_t
        key = torch.cat((
            # caption 
            self.key_text(self.text_norm(xf)) + (1 - text_cond_type) * -1000000,  
            # retrieve
            self.key_retr(self.retr_norm1(re_feat_key)) + (1 - retr_cond_type) * -1000000,  
            # motion
            self.key_motion(self.norm(x)) + (1 - src_mask) * -1000000  
        ), dim=1)
        
        '''------------- <VALUE> ------------'''
        re_feat_value = re_motion.reshape(B, -1, D)  # (1, k*n, latent_dim), R_m
        value = torch.cat((
            # caption 
            self.value_text(self.text_norm(xf)) * text_cond_type,
            # retrieve
            self.value_retr(self.retr_norm2(re_feat_value)) * retr_cond_type,
            # motion
            self.value_motion(self.norm(x)) * src_mask,
        ), dim=1)
        

        '''------------ <ATTENTION> ------------'''
        query = F.softmax(query.view(B, T, H, -1), dim=-1)   # (b, n, h, h_dim)  
        key = F.softmax(key.view(B, N, H, -1), dim=1)  # (b, N_all, latent_dim) 
        value = value.view(B, N, H, -1)   # (b, N_all, latent_dim) 

        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)  
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)  # (b, n, latent_dim) 

        return y    

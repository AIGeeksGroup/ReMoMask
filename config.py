# BMM 
global_bimoco_config = {
    "motion_embedding_dims":  512,
    "text_embedding_dims":  768,
    "projection_dims": 512
}

# Momentum 
global_momentum_config = {
    'embed_dim' : 512,
    'queue_size' : 65536,   # must can be divided by RAG's bacth_size
    'momentum' : 0.99,
}


# RAG
retriever_cfg=dict(
        motion_codebook_size=512,
        database_path="database",   
        tmr_model_path="Part_TMR/checkpoints/exp1/HumanML3D",  
        device = "cuda",
)


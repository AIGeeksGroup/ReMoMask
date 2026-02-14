# <img src="./assets/remomask_logo.png" alt="logo" width="30"/> ReMoMask: Retrieval-Augmented Masked Motion Generation<br>

This is the official repository for the paper:
> **ReMoMask: Retrieval-Augmented Masked Motion Generation**
>
> Zhengdao Li\*, Siheng Wang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> [Paper](https://arxiv.org/abs/2508.02605) | [Website](https://aigeeksgroup.github.io/ReMoMask) | [Model](https://huggingface.co/lycnight/ReMoMask/blob/main/README.md) | [HF Paper](https://huggingface.co/papers/2508.02605)


<!-- <video> -->

# ✏️ Citation

```
@article{li2025remomask,
  title={ReMoMask: Retrieval-Augmented Masked Motion Generation},
  author={Li, Zhengdao and Wang, Siheng and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2508.02605},
  year={2025}
}
```

---

# 👋 Introduction
Retrieval-Augmented Text-to-Motion (RAG-T2M) models have demonstrated superior performance over conventional T2M approaches, particularly in handling uncommon and complex textual descriptions by leveraging external motion knowledge. Despite these gains, existing RAG-T2M models remain limited by two closely related factors: (1) coarse-grained text-motion retrieval that overlooks the hierarchical structure of human motion, and (2) underexplored mechanisms for effectively fusing retrieved information into the generative process. In this work, we present **ReMoMask**, a structure-aware RAG framework for text-to-motion generation that addresses these limitations. To improve retrieval, we propose **Hierarchical Bidirectional Momentum** (HBM) Contrastive Learning, which employs dual contrastive objectives to jointly align global motion semantics and fine-grained part-level motion features with text. To address the fusion gap, we first conduct a systematic study on motion representations and information fusion strategies in RAG-T2M, revealing that a 2D motion representation combined with cross-attention-based fusion yields superior performance. Based on these findings, we design **Semantic Spatial-Temporal Attention** (SSTA), a motion-tailored fusion module that more effectively integrates retrieved motion knowledge into the generative backbone. Extensive experiments on HumanML3D, KIT-ML, and SnapMoGen demonstrate that ReMoMask consistently outperforms prior methods on both text-motion retrieval and text-to-motion generation benchmarks.



![framework](./assets/framework.png)

## TODO List

- [x] Upload our paper to arXiv and build project pages.
- [x] Upload the code.
- [x] Release TMR model.
- [x] Release T2M model.

# 🤗 Prerequisite
<details> 
<summary>details</summary>
  
## Environment
```bash
conda create -n remomask python=3.10
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
conda activate remomask
```
We tested our environment on both A800 and H20.

## Dependencies
### 1. pretrained models
Dwonload the models from [HuggingFace](https://huggingface.co/lycnight/ReMoMask), and place them like:

```
remomask_models.zip
    ├── checkpoints/              #   Evaluation Models and Gloves
    ├── Part_TMR/
    │   └── checkpoints/        # RAG pretrained checkpoints
    ├── logs/                   # T2M pretrained checkpoints
    ├── database/               # RAG database
    └── ViT-B-32.pt            # CLIP model
```

### 2. Prepare training dataset 
Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then place the result dataset to `./dataset/HumanML3D`.
</details>

# 🚀 Demo
<details> 
<summary>details</summary>
  
```bash
python demo.py \
    --gpu_id 0 \
    --ext exp_demo \
    --text_prompt "A person is playing the drum set." \
    --checkpoints_dir logs \
    --dataset_name humanml3d \
    --mtrans_name pretrain_mtrans \
    --rtrans_name pretrain_rtrans
# change pretrain_mtrans and pretrain_rtrans to your mtrans and rtrans after your training done
```
explanation:
* `--repeat_times`: number of replications for generation, default `1`.
* `--motion_length`: specify the number of poses for generation.

output will be in `./outputs/`
</details> 


# 🛠️ Train your own models
<details>
<summary>details</summary>
  
## Stage1: train a Motion Retriever
```bash
python Part_TMR/scripts/train.py \
    device=cuda:0 \
    train=train \
    dataset.train_split_filename=train.txt \
    exp_name=exp \
    train.optimizer.motion_lr=1.0e-05 \
    train.optimizer.text_lr=1.0e-05 \
    train.optimizer.head_lr=1.0e-05
# change the exp_name to your rag name
```
then build a rag database for training t2m model:
```bash
python build_rag_database.py \
    --config-name=config \
    device=cuda:0 \
    train=train \
    dataset.train_split_filename=train.txt \
    exp_name=exp_for_mtrans
```
you will get `./database`


##  Stage2: train a Retrieval Augmented Mask Model

### tarin a 2D RVQ-VAE Quantizer
```bash
bash run_rvq.sh \
    vq \
    0 \
    humanml3d \
    --batch_size 256 \
    --num_quantizers 6 \
    --max_epoch 50 \
    --quantize_dropout_prob 0.2 \
    --gamma 0.1 \
    --code_dim2d 1024 \
    --nb_code2d 256
# vq means the save dir
# 0 means gpu_0
# humanml3d means dataset
# change the vq_name to your vq name
```

### train a 2D Retrieval-Augmented Mask Transformer
```bash
bash run_mtrans.sh \
    mtrans \
    1 \
    0 \
    11247 \
    humanml3d \
    --vq_name pretrain_vq \
    --batch_size 64 \
    --max_epoch 2000 \
    --attnj \
    --attnt \
    --latent_dim 512 \
    --n_heads 8 \
    --train_split train.txt \
    --val_split val.txt
# 1 means using one gpu
# 0 means using gpu_0
# 11247 means ddp master port
# change the mtrans to your mtrans name
```


### train a 2D Retrieval-Augmented Residual Transformer
```bash
bash run_rtrans.sh \
    rtrans \
    2 \
    humanml3d \
    --batch_size 64 \
    --vq_name pretrain_vq \
    --cond_drop_prob 0.01 \
    --share_weight \
    --max_epoch 2000 \
    --attnj \
    --attnt
# here, 2 means cuda:0,1
# --vq_name: the vq model you want to use
# change the rtrans to your vq rtrans
```

</details>



# 💪 Evalution
<details>
<summary>details</summary>
  
## Evaluate the RAG  
```bash
python Part_TMR/scripts/test.py \
    device=cuda:0 \
    train=train \
    exp_name=exp_pretrain
# change exp_pretrain to your rag model
```


## Evaluate the T2M

### 1. Evaluate the 2D RVQ-VAE Quantizer
```bash
python eval_vq.py \
--gpu_id 0 \
--name pretrain_vq \
--dataset_name humanml3d \
--ext eval \
--which_epoch net_best_fid.tar
# change pretrain_vq to your vq
```

### 2. Evaluate the 2D Retrieval-Augmented Masked Transformer
```bash
python eval_mask.py \
    --dataset_name humanml3d \
    --mtrans_name pretrain_mtrans \
    --gpu_id 0 \
    --cond_scale 4 \
    --time_steps 10 \
    --ext eval \
    --repeat_times 1 \
    --which_epoch net_best_fid.tar
# change pretrain_mtrans to your mtrans
```


### 3. Evaluate the 2D Residual Transformer
HumanML3D:
```bash
python eval_res.py \
    --gpu_id 0 \
    --dataset_name humanml3d \
    --mtrans_name pretrain_mtrans \
    --rtrans_name pretrain_rtrans \
    --cond_scale 4 \
    --time_steps 10 \
    --ext eval \
    --which_ckpt net_best_fid.tar \
    --which_epoch fid \
    --traverse_res
# change pretrain_mtrans and pretrain_rtrans to your mtrans and rtrans
```
</details>



# 🤖 Visualization
<details>
<summary>details</summary>
  
## 1. download and set up blender
<details>
<summary>details</summary>
You can download the blender from [instructions](https://www.blender.org/download/lts/2-93/). Please install exactly this version. For our paper, we use `blender-2.93.18-linux-x64`. 
> 
### a. unzip it:
```bash
tar -xvf blender-2.93.18-linux-x64.tar.xz
```

### b. check if you have installed the blender successfully or not:
```bash
cd blender-2.93.18-linux-x64
./blender --background --version
```
you should see: `Blender 2.93.18 (hash cb886axxxx built 2023-05-22 23:33:27)`
```bash
./blender --background --python-expr "import sys; import os; print('\nThe version of python is ' + sys.version.split(' ')[0])"
```
you should see: `The version of python is 3.9.2`

### c. get the blender-python path
```bash
./blender --background --python-expr "import sys; import os; print('\nThe path to the installation of python is\n' + sys.executable)"
```
you should see: `	The path to the installation of python is /xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9s`

### d. install pip for blender-python
```bash
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m ensurepip --upgrade
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install --upgrade pip
```

### e. prepare env for blender-python
```bash 
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install numpy==2.0.2
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install matplotlib==3.9.4
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install hydra-core==1.3.2
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install hydra_colorlog==1.2.0
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install moviepy==1.0.3
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install shortuuid==1.0.13
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install natsort==8.4.0
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install pytest-shutil==1.8.1
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install tqdm==4.67.1
/xxx/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install tqdm==1.17.0
```
</details>


## 2. calulate SMPL mesh:
```bash
python -m fit --dir new_test_npy --save_folder new_temp_npy --cuda cuda:0
```

## 3. render to video or sequence
```bash
/xxx/blender-2.93.18-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --dir=test_npy --mode=video --joint_type=HumanML3D
```
- `--mode=video`: render to mp4 video
- `--mode=sequence`: render to a png image, calle sequence.

</details>

# 👍 Acknowlegements
We sincerely thank the open-sourcing of these works where our code is based on:

[MoMask](https://github.com/EricGuo5513/momask-codes),
[MoGenTS](https://github.com/weihaosky/mogents),
[ReMoDiffuse](https://github.com/mingyuan-zhang/ReMoDiffuse),
[MDM](https://github.com/GuyTevet/motion-diffusion-model),
[TMR](https://github.com/Mathux/TMR),
[ReMoGPT](https://ojs.aaai.org/index.php/AAAI/article/view/33044)

## 🔒 License
This code is distributed under an [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.

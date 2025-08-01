import codecs as cs
import random
from os.path import join as pjoin

# import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils import data
from tqdm import tqdm

from .utils import whole2parts


class TextMotionDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        mean,
        std,
        split_file,
        eval_mode=False,
        fps=None,
    ):
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_motion_length = cfg.dataset.max_motion_length
        self.fps = fps

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(cfg.dataset.motion_dir, name + ".npy"))
                if len(motion.shape) != 2:
                    continue
                if np.isnan(motion).any():
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(cfg.dataset.text_dir, name + ".txt")) as f:
                    for index, line in enumerate(f.readlines()):
                        if eval_mode and index >= 1:
                            continue
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[
                                int(f_tag * cfg.dataset.fps) : int(
                                    to_tag * cfg.dataset.fps
                                )
                            ]

                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                            )
                            while new_name in data_dict:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                            data_dict[new_name] = {
                                "motion": n_motion,
                                "length": len(n_motion),
                                "text": [text_dict],
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list

        for key, item in tqdm(data_dict.items()):
            motion = data_dict[key]["motion"]

            motion = (motion - self.mean) / self.std
            motion = np.array(motion, dtype=np.float32)

            data_dict[key]["pre_motion"] = motion
            data_dict[key]["length"] = motion.shape[0]

    def real_len(self):
        return len(self.data_dict)

    def _subsample_to_20fps(self, orig_ft, orig_fps):
        T, n_j, _ = orig_ft.shape
        out_fps = 20.0
        # Matching the sub-sampling used for rendering
        if int(orig_fps) % int(out_fps):
            sel_fr = np.floor(orig_fps / out_fps * np.arange(int(out_fps))).astype(int)
            n_duration = int(T / int(orig_fps))
            t_idxs = []
            for i in range(n_duration):
                t_idxs += list(i * int(orig_fps) + sel_fr)
            if int(T % int(orig_fps)):
                last_sec_frame_idx = n_duration * int(orig_fps)
                t_idxs += [
                    x + last_sec_frame_idx for x in sel_fr if x + last_sec_frame_idx < T
                ]
        else:
            t_idxs = np.arange(0, T, orig_fps / out_fps, dtype=int)

        ft = orig_ft[t_idxs, :, :]
        return ft

    def __len__(self):
        return self.real_len() * self.cfg.dataset.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["pre_motion"], data["length"], data["text"]
        # Randomly select a caption
        if self.eval_mode:
            caption = text_list[0]["caption"]
        else:
            text_data = random.choice(text_list)
            caption = text_data["caption"]

        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            idx = (
                random.randint(0, len(motion) - max_motion_length)
                if not self.eval_mode
                else 0
            )
            motion = motion[idx : idx + max_motion_length]
            m_length = max_motion_length
        else:
            if self.cfg.preprocess.padding:
                padding_len = max_motion_length - m_length
                D = motion.shape[1]
                padding_zeros = np.zeros((padding_len, D), dtype=np.float32)
                motion = np.concatenate((motion, padding_zeros), axis=0)

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(
            motion,
            mode="t2m" if self.cfg.dataset.dataset_name == "HumanML3D" else "kit",
        )

        return Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, caption, m_length, item




class Text2CoarseMotionDataset(TextMotionDataset):
    def __init__(
        self,
        cfg,
        mean,
        std,
        split_file,
        eval_mode=False,
        fps=None,
    ):
        super().__init__(cfg, mean, std, split_file, eval_mode, fps)

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["pre_motion"], data["length"], data["text"]
        # Randomly select a caption
        if self.eval_mode:
            caption = text_list[0]["caption"]
        else:
            text_data = random.choice(text_list)
            caption = text_data["caption"]

        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            idx = (
                random.randint(0, len(motion) - max_motion_length)
                if not self.eval_mode
                else 0
            )
            motion = motion[idx : idx + max_motion_length]
            m_length = max_motion_length
        else:
            if self.cfg.preprocess.padding:
                padding_len = max_motion_length - m_length
                D = motion.shape[1]
                padding_zeros = np.zeros((padding_len, D), dtype=np.float32)
                motion = np.concatenate((motion, padding_zeros), axis=0)

        # Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(
        #     motion,
        #     mode="t2m" if self.cfg.dataset.dataset_name == "HumanML3D" else "kit",
        # )

        return motion, caption, m_length, item





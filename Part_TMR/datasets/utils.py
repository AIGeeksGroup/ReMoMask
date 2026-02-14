"""
# This code is based on https://github.com/qrzou/ParCo
"""

import codecs as cs
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


def whole2parts(motion, mode="t2m", window_size=None):
    '''
    它将 完整格式的动作数据（如 263 ）还原成按部位划分的结构化动作数据（Root、四肢、Backbone等六部分）。
    输入：
        - motion: (t, raw_motion_dim)  # numpy.ndarray
    输出:
        - list[part_motion]
            - 其中的 part_motion: (t, part_motion_dim)
    '''

    # motion.shape: (224, 263)  # (nframes, raw_motion_dim)
    if type(motion) == np.ndarray:
        aug_data = torch.from_numpy(motion).float()
    else:
        aug_data = motion
    # aug_data.shape: (224, 263)  # (nframes, raw_motion_dim)

    # motion
    if mode == "t2m":
        # 263-dims motion is actually an augmented motion representation
        # split the 263-dims data into the separated augmented data form:
        #    root_data, ric_data, rot_data, local_vel, feet
        # (nframes, 263)

        # ------------- part 1 --------------
        joints_num = 22
        s = 0  # start
        e = 4  # end
        root_data = aug_data[:, s:e]  # [seg_len-1, 4]  # torch.Size([224, 4])  # 前四维是根节点数据
        
        # ------------- part 2 --------------
        s = e   # 4
        e = e + (joints_num - 1) * 3    # 4 + (22 - 1) * 3 = 67
        ric_data = aug_data[
            :, s:e
        ]  # [seq_len, (joints_num-1)*3]. (joints_num - 1) means the 0th joint is dropped.
        # torch.Size([224, 63])  # 接下来63维是关节位置数据

         # ------------- part 3 --------------
        s = e   # 67
        e = e + (joints_num - 1) * 6       # 67 + (22 - 1) * 6 = 193
        rot_data = aug_data[:, s:e]  # [seq_len, (joints_num-1) *6]   
        # torch.Size([224, 126])  # 接下来126维是关节旋转数据

        # ------------- part 4 --------------
        s = e  # 193
        e = e + joints_num * 3  # 193 + 22 * 3 = 259
        local_vel = aug_data[:, s:e]  # [seq_len-1, joints_num*3]
        # torch.Size([224, 66])  # 接下来66维是关节速度数据

         # ------------- part 5 --------------
        s = e  # 259
        e = e + 4   # 259 + 4 = 263
        feet = aug_data[:, s:e]  # [seg_len-1, 4]
        # torch.Size([224, 4])  # 最后四维是脚部数据

        '''
        root_data : torch.Size([224, 4])
        ric_data : torch.Size([224, 63])
        rot_data : torch.Size([224, 126])
        local_vel : torch.Size([224, 66])
        feet : torch.Size([224, 4])
        '''

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(torch.int64)  # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(torch.int64)  # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(torch.int64)  # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(torch.int64)  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(torch.int64)  # left arm

        nframes = root_data.shape[0]    # 224
        if window_size is not None:     # None
            assert nframes == window_size

        # The original shape of root_data and feet
        # root_data: (nframes, 4)
        # feet: (nframes, 4)
        ric_data = ric_data.reshape(nframes, -1, 3)  # (nframes, joints_num - 1, 3) # torch.Size([224, 63]) -> torch.Size([224, 21, 3])
        rot_data = rot_data.reshape(nframes, -1, 6)  # (nframes, joints_num - 1, 6) # torch.Size([224, 126]) -> torch.Size([224, 21, 6])
        local_vel = local_vel.reshape(nframes, -1, 3)  # (nframes, joints_num, 3)   # local_vel : torch.Size([224, 66]) -> torch.Size([224, 22, 3])

        # root_data.shape:   torch.Size([224, 4])
        # local_vel[:, 0, :].shape:    torch.Size([224, 3])
        root_data = torch.cat(
            [root_data, local_vel[:, 0, :]], dim=1
        )  # (nframes, 4+3=7)   # torch.Size([224, 7])

        # ---------------------------- part 1 --------------------------------
        R_L = torch.cat(
            [
                ric_data[:, R_L_idx - 1, :],  # torch.Size([224, 4, 3])
                rot_data[:, R_L_idx - 1, :],  # torch.Size([224, 4, 6])
                local_vel[:, R_L_idx, :],     # torch.Size([224, 4, 3])
            ],
            dim=2,
        )  # (nframes, 4, 3+6+3=12)  # torch.Size([224, 4, 12])

        # ---------------------------- part 2 --------------------------------
        L_L = torch.cat(
            [
                ric_data[:, L_L_idx - 1, :],     # torch.Size([224, 4, 3])
                rot_data[:, L_L_idx - 1, :],     # torch.Size([224, 4, 6])
                local_vel[:, L_L_idx, :],        # torch.Size([224, 4, 3])
            ],
            dim=2,
        )  # (nframes, 4, 3+6+3=12)     # torch.Size([224, 4, 12])

        # ---------------------------- part 3 --------------------------------
        B = torch.cat(
            [
                ric_data[:, B_idx - 1, :],  # torch.Size([224, 5, 3])
                rot_data[:, B_idx - 1, :],  # torch.Size([224, 5, 6])
                local_vel[:, B_idx, :],     # torch.Size([224, 5, 3])
            ],
            dim=2,
        )  # (nframes, 5, 3+6+3=12)     #   torch.Size([224, 5, 12])

        # ---------------------------- part 4 --------------------------------
        R_A = torch.cat(
            [
                ric_data[:, R_A_idx - 1, :],  # torch.Size([224, 5, 3])
                rot_data[:, R_A_idx - 1, :],  # torch.Size([224, 5, 6])
                local_vel[:, R_A_idx, :],     # torch.Size([224, 5, 3])
            ],
            dim=2,
        )  # (nframes, 5, 3+6+3=12)   # torch.Size([224, 5, 12])

        # ---------------------------- part 5 --------------------------------
        L_A = torch.cat(
            [
                ric_data[:, L_A_idx - 1, :],   # torch.Size([224, 5, 3])
                rot_data[:, L_A_idx - 1, :],   # torch.Size([224, 5, 6])
                local_vel[:, L_A_idx, :],      # torch.Size([224, 5, 3])
            ],
            dim=2,
        )  # (nframes, 5, 3+6+3=12)   # torch.Size([224, 5, 12])


        # ---------------------------- part 6 (Backbone)--------------------------------
        Root = root_data  # (nframes, 4+3=7)    # torch.Size([224, 7])
        
        R_Leg = torch.cat(
            [R_L.reshape(nframes, -1), feet[:, 2:]], dim=1
        )  # (nframes, 4*12+2=50)
        # torch.Size([224, 50]) 

        L_Leg = torch.cat(
            [L_L.reshape(nframes, -1), feet[:, :2]], dim=1
        )  # (nframes, 4*12+2=50)
        # torch.Size([224, 50])

        Backbone = B.reshape(nframes, -1)  # (nframes, 5*12=60)
        # torch.Size([224, 60])

        R_Arm = R_A.reshape(nframes, -1)  # (nframes, 5*12=60)
        # torch.Size([224, 60])

        L_Arm = L_A.reshape(nframes, -1)  # (nframes, 5*12=60)
        # torch.Size([224, 60])

    elif mode == "kit":
        # 251-dims motion is actually an augmented motion representation
        # split the 251-dims data into the separated augmented data form:
        #    root_data, ric_data, rot_data, local_vel, feet
        joints_num = 21
        s = 0  # start
        e = 4  # end
        root_data = aug_data[:, s:e]  # [seg_len-1, 4]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[
            :, s:e
        ]  # [seq_len, (joints_num-1)*3]. (joints_num - 1) means the 0th joint is dropped.
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[:, s:e]  # [seq_len, (joints_num-1) *6]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]  # [seq_len-1, joints_num*3]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]  # [seg_len-1, 4]

        # move the root joint 0-th out of belowing parts
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(torch.int64)  # right leg
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(torch.int64)  # left leg
        B_idx = torch.Tensor([1, 2, 3, 4]).to(torch.int64)  # backbone
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(torch.int64)  # right arm
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(torch.int64)  # left arm

        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size

        # The original shape of root_data and feet
        # root_data: (nframes, 4)
        # feet: (nframes, 4)
        ric_data = ric_data.reshape(nframes, -1, 3)  # (nframes, joints_num - 1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)  # (nframes, joints_num - 1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)  # (nframes, joints_num, 3)

        root_data = torch.cat(
            [root_data, local_vel[:, 0, :]], dim=1
        )  # (nframes, 4+3=7)
        R_L = torch.cat(
            [
                ric_data[:, R_L_idx - 1, :],
                rot_data[:, R_L_idx - 1, :],
                local_vel[:, R_L_idx, :],
            ],
            dim=2,
        )  # (nframes, 4, 3+6+3=12)
        L_L = torch.cat(
            [
                ric_data[:, L_L_idx - 1, :],
                rot_data[:, L_L_idx - 1, :],
                local_vel[:, L_L_idx, :],
            ],
            dim=2,
        )  # (nframes, 4, 3+6+3=12)
        B = torch.cat(
            [
                ric_data[:, B_idx - 1, :],
                rot_data[:, B_idx - 1, :],
                local_vel[:, B_idx, :],
            ],
            dim=2,
        )  # (nframes, 5, 3+6+3=12)
        R_A = torch.cat(
            [
                ric_data[:, R_A_idx - 1, :],
                rot_data[:, R_A_idx - 1, :],
                local_vel[:, R_A_idx, :],
            ],
            dim=2,
        )  # (nframes, 5, 3+6+3=12)
        L_A = torch.cat(
            [
                ric_data[:, L_A_idx - 1, :],
                rot_data[:, L_A_idx - 1, :],
                local_vel[:, L_A_idx, :],
            ],
            dim=2,
        )  # (nframes, 5, 3+6+3=12)

        Root = root_data  # (nframes, 4+3=7)
        R_Leg = torch.cat(
            [R_L.reshape(nframes, -1), feet[:, 2:]], dim=1
        )  # (nframes, 4*12+2=50)
        L_Leg = torch.cat(
            [L_L.reshape(nframes, -1), feet[:, :2]], dim=1
        )  # (nframes, 4*12+2=50)
        Backbone = B.reshape(nframes, -1)  # (nframes, 5*12=60)
        R_Arm = R_A.reshape(nframes, -1)  # (nframes, 5*12=60)
        L_Arm = L_A.reshape(nframes, -1)  # (nframes, 5*12=60)

    else:
        raise Exception()

    # [Root: (224, 7), R_Leg: (224, 50), L_Leg: (224, 50), Backbone: (224, 60), R_Arm: (224, 60), L_Arm: (224, 60)]
    return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]


def parts2whole(parts, mode="t2m", shared_joint_rec_mode="Avg"):
    assert isinstance(parts, list)

    if mode == "t2m":
        # Parts to whole. (7, 50, 50, 60, 60, 60) ==> 263
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 22
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(device, dtype=torch.int64)  # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(device, dtype=torch.int64)  # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(
            device, dtype=torch.int64
        )  # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(
            device, dtype=torch.int64
        )  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(
            device, dtype=torch.int64
        )  # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[:, 0, :] = Root[:, 4:]

        else:
            R_L = R_Leg[..., :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(bs, nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip(
            [R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]
        ):
            # rec_ric_data[:, idx - 1, :] = part[:, :, :3]
            # rec_rot_data[:, idx - 1, :] = part[:, :, 3:9]
            # rec_local_vel[:, idx, :] = part[:, :, 9:]

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 9th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 9

        if shared_joint_rec_mode == "L_Arm":
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == "R_Arm":
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == "Backbone":
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == "Avg":
            rec_ric_data[..., idx - 1, :] = (
                L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]
            ) / 3
            rec_rot_data[..., idx - 1, :] = (
                L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]
            ) / 3
            rec_local_vel[..., idx, :] = (
                L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]
            ) / 3

        else:
            raise Exception()

        # Concate them to 263-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=1,
            )

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=2,
            )

    elif mode == "kit":

        # Parts to whole. (7, 62, 62, 48, 48, 48) ==> 251
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 21
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(
            device, dtype=torch.int64
        )  # right leg
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(
            device, dtype=torch.int64
        )  # left leg
        B_idx = torch.Tensor([1, 2, 3, 4]).to(device, dtype=torch.int64)  # backbone
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(device, dtype=torch.int64)  # right arm
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(device, dtype=torch.int64)  # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[:, 0, :] = Root[:, 4:]

        else:
            R_L = R_Leg[..., :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(bs, nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip(
            [R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]
        ):

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 3-th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 3

        if shared_joint_rec_mode == "L_Arm":
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == "R_Arm":
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == "Backbone":
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == "Avg":
            rec_ric_data[..., idx - 1, :] = (
                L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]
            ) / 3
            rec_rot_data[..., idx - 1, :] = (
                L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]
            ) / 3
            rec_local_vel[..., idx, :] = (
                L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]
            ) / 3

        else:
            raise Exception()

        # Concate them to 251-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=1,
            )

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=2,
            )

    else:
        raise Exception()

    return rec_data


def get_each_part_vel(parts, mode="t2m"):
    assert isinstance(parts, list)

    if mode == "t2m":
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(
                bs, nframes, -1
            )  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [
            Root_vel,
            R_Leg_vel,
            L_Leg_vel,
            Backbone_vel,
            R_Arm_vel,
            L_Arm_vel,
        ]

    elif mode == "kit":
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(
                bs, nframes, -1
            )  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [
            Root_vel,
            R_Leg_vel,
            L_Leg_vel,
            Backbone_vel,
            R_Arm_vel,
            L_Arm_vel,
        ]

    else:
        raise Exception()

    return parts_vel_list  # [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]


from typing import Any, List, Tuple, Union


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax. From stylegan2-ADA"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

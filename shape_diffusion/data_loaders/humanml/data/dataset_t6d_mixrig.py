import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from argparse import Namespace
from scipy.sparse import csr_matrix
from smplx.lbs import batch_rigid_transform as batch_rigid_transform_torch
from scipy.spatial.transform import Rotation as rot

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt

def read_obj(path):
    """
    read verts and faces from obj file. This func will convert quad mesh to triangle mesh
    """
    with open(path) as f:
        lines = f.read().splitlines()
    verts = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            part = line.split(' ')[1:]
            if len(part)!=3:
                continue
            vert = np.array([float(k) if k!='' else 0.0 for k in part])
            verts.append(vert)
        elif line.startswith('f '):
            try:
                if "//" in line:
                    onef = np.array([int(k.split("//")[0]) for k in line.split(' ')[1:]])
                else:
                    onef = np.array([int(k.split("/")[0]) for k in line.split(' ')[1:]])
            except ValueError:
                print(line)
                continue
            if len(onef) == 4:
                faces.append(onef[[0, 1, 2]])
                faces.append(onef[[0, 2, 3]])
            elif len(onef) > 4:
                pass
            else:
                faces.append(onef)
    if len(faces) == 0:
        return np.stack(verts), None
    else:
        return np.stack(verts), np.stack(faces)-1

def batch_rigid_transform(rot_mats, joints, parents):
    posed_joints, rel_transforms = batch_rigid_transform_torch(
        torch.from_numpy(rot_mats).float(),
        torch.from_numpy(joints).float(),
        torch.from_numpy(parents).long(),
    )
    return posed_joints.numpy(), rel_transforms.numpy()


def lbs_batch(v, rotations, J, parents, weights, input_mat=False):
    """
    Args:
        v: (B, V, 3)
        rotations: (B, J, 3), rotation vector
        J: (B, J, 3), joint positions
        parents: (J), kinematic chain indicator
        weights: (B, V, J), skinning weights

    Returns:
        articulated vertices: (B, V, 3)
    """
    B, num_joints, _ = rotations.shape
    v = np.tile(v, (B, 1, 1))
    J = np.tile(J, (B, 1, 1))
    if input_mat:
        rot_mats = rotations.reshape([B, num_joints, 3, 3])
    else:
        rot_mats = (
            rot.from_rotvec(rotations.reshape([-1, 3]))
            .as_matrix()
            .reshape([B, num_joints, 3, 3])
        )
        # rot_mats = rot.from_euler("XYZ", rotations.reshape([-1, 3])).as_matrix().reshape([B, num_joints, 3, 3])
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    # W = np.tile(weights[None], [B, 1, 1])

    W = np.tile(weights, [B, 1, 1])
    T = np.einsum("bnj,bjpq->bnpq", W, A)

    R = T[:, :, :3, :3]
    t = T[:, :, :3, 3]
    verts = np.einsum("bnpq,bnq->bnp", R, v) + t
    # verts = np.einsum("bnpq,bnq->bnp", R, v)
    return verts, J_transformed, T


class Text2MotionDatasetTrans(data.Dataset):
    def __init__(
        self, opt, mean, std, split_file, w_vectorizer, mixamo_dir, rignet_dir, mode
    ):
        self.opt = opt
        self.mixamo_dir = mixamo_dir
        self.rignet_dir = rignet_dir
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == 't6d_mixrig' else 24

        self.mode = mode

        self.MIXAMO_JOINTS = [
            "Hips",
            "Spine",
            "Spine1",
            "Spine2",
            "Neck",
            "Head",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "LeftToeBase",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "RightToeBase",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
        ]

        self.FOCUS_HANDLES= [2,11,26, 13,8,19]

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        id_list = id_list[:100]  # debug

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            # try:
            motion_path = pjoin(opt.motion_dir, name + '.npz')
            theta_path = pjoin(opt.theta_dir, name + '.npz')
            if not os.path.exists(motion_path):
                continue
            motion = self.load_motion(motion_path)
            theta = self.load_theta(theta_path)
            if (len(motion)) < min_motion_len or (len(motion) >= 200):
                continue
            text_data = []
            flag = False
            with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        try:
                            n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                            n_theta = theta[int(f_tag * 20) : int(to_tag * 20)]
                            if (len(n_motion)) < min_motion_len or (
                                len(n_motion) >= 200
                            ):
                                continue
                            new_name = (
                                random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            )
                            while new_name in data_dict:
                                new_name = (
                                    random.choice('ABCDEFGHIJKLMNOPQRSTUVW')
                                    + '_'
                                    + name
                                )
                            data_dict[new_name] = {
                                'motion': n_motion,
                                'theta': n_theta,
                                'length': len(n_motion),
                                'text': [text_dict],
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, name)
                            # break

            if flag:
                data_dict[name] = {
                    'motion': motion,
                    'theta': theta,
                    'length': len(motion),
                    'text': text_data,
                }
                new_name_list.append(name)
                length_list.append(len(motion))
            # except:
            #     pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.mean = mean  # 1 41, 7
        self.std = std  # 1 41, 7
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

        # load mixamo
        (
            self.feat_dict,
            self.hand_dict,
            self.char_name_lst,
            self.joint_dict,
            self.parents_dict,
            self.handle_weights_dict,
            self.joint_names_dict,
            self.weight_dict,
            self.score_dict,
            self.skin_dict
        ) = self.load_mixamo(mixamo_dir)
        self.mixamo_char_num = len(self.char_name_lst)

        self.load_mixamo_motion()

        # load rignet
        # self.feat_dict_rig, self.hand_dict_rig, self.char_name_lst_rig,\
        #       self.joint_dict_rig, self.parents_dict_rig, self.handle_weights_dict_rig, self.weight_dict_rig, self.score_dict_rig, self.skin_dict_rig = self.load_rignet(rignet_dir)
        # self.rignet_char_num = len(self.char_name_lst_rig)

    def load_motion(self, motion_path):
        motion = np.load(motion_path)
        body_motion = motion["body_motion"]  # T 40 9 (3 translation + 6d)
        root_rotation = motion["root_rotation"]  # T 6 (6d)
        root_veloc = motion["root_veloc"]  # T 3
        root_pos = motion["root_pos"]  # T 3

        T = root_rotation.shape[0]
        root_rotation = root_rotation[:, None, :]
        root_veloc = root_veloc[:, None, :]
        root_motion = np.concatenate([root_veloc, root_rotation], axis=2)  # T 1 9

        motion = np.concatenate([body_motion, root_motion], axis=1)  # T 41 9

        return motion

    def load_theta(self, theta_path):
        data = np.load(theta_path)

        root_orient = data["root_orient"]
        pose_body = data["pose_body"].reshape(root_orient.shape[0], 21, 3)
        pose_body = np.concatenate(
            [root_orient[:, None, :], pose_body], axis=1
        )  # T 22 3

        return pose_body

    def load_mixamo(self, mixamo_dir):
        file_folder = "features_temp"
        features_dir = pjoin(mixamo_dir, file_folder)
        features_lst = sorted(
            [
                feature_name
                for feature_name in os.listdir(features_dir)
                if "featurex4" in feature_name
            ]
        )
        handle_lst = sorted(
            [
                handle_name
                for handle_name in os.listdir(features_dir)
                if "handle" in handle_name
            ]
        )
        score_lst = sorted(
            [
                score_name
                for score_name in os.listdir(features_dir)
                if "score" in score_name
            ]
        )
        score_skin_lst = sorted(
            [
                score_skin_name
                for score_skin_name in os.listdir(features_dir)
                if "skinning" in score_skin_name
            ]
        )
        char_name_lst = [feature_name[:-14] for feature_name in features_lst]

        # features_lst = features_lst[0:2]
        # handle_lst = handle_lst[0:2]
        # score_lst = score_lst[0:2]
        # score_skin_lst = score_skin_lst[0:2]

        feat_dict = {}
        for i, feat in enumerate(features_lst):
            feat_path = pjoin(features_dir, feat)
            feat_np = np.load(feat_path)[None,]  # 1 256
            feat_dict[char_name_lst[i]] = feat_np

        handle_dict = {}
        for i, hand in enumerate(handle_lst):
            hand_path = pjoin(features_dir, hand)
            hand_np = np.load(hand_path)[None,]  # 1 40 3
            handle_dict[char_name_lst[i]] = hand_np

        score_dict = {}
        for i, score in enumerate(score_lst):
            score_path = pjoin(features_dir, score)
            score_np = np.load(score_path)[None,]  # 1 N 40
            score_dict[char_name_lst[i]] = score_np


        score_skin_dict = {}
        for i, skin in enumerate(score_skin_lst):
            skin_path = pjoin(features_dir, skin)
            skin_np = np.load(skin_path)[None,]  # 1 N 40
            score_skin_dict[char_name_lst[i]] = skin_np

        joint_dict = {}
        weight_dict = {}
        parents_dict = {}
        handle_weights_dict = {}
        joint_names_dict = {}

        for idx in char_name_lst:
            c = np.load(os.path.join(self.mixamo_dir, 'npz', idx + '.npz'))
            # v = c["v"]*100
            # center = (np.max(v, 0, keepdims=True) + np.min(v, 0, keepdims=True)) / 2
            # scale = np.max(v[:, 1], 0) - np.min(v[:, 1], 0)
            center = np.load(
                os.path.join(self.mixamo_dir, file_folder, idx + '_center.npy')
            )
            scale = np.load(
                os.path.join(self.mixamo_dir, file_folder, idx + '_scale.npy')
            )
            joints = c["joints"]
            joints = (joints - (100*center)) / (100*scale)

            joint_dict[idx] = joints
            parents_dict[idx] = c["parents"]
            weight_dict[idx] = csr_matrix(
                (c["weights_data"], (c["weights_rows"], c["weights_cols"])),
                shape=c["weights_shape"],
                dtype=np.float32,
            ).toarray()

            handle_weight = np.einsum(
                "abc,bd->acd", score_dict[idx], weight_dict[idx]
            )  # 1 40 22
            handle_weights_dict[idx] = handle_weight

            joint_names_dict[idx] = c["joint_names"].tolist()

        return (
            feat_dict,
            handle_dict,
            char_name_lst,
            joint_dict,
            parents_dict,
            handle_weights_dict,
            joint_names_dict,
            weight_dict,
            score_dict,
            score_skin_dict
        )

    def load_rignet(self, rignet_dir):
        features_dir = pjoin(rignet_dir, "features_30part")
        features_lst = sorted(
            [
                feature_name
                for feature_name in os.listdir(features_dir)
                if "featurex4" in feature_name
            ]
        )
        handle_lst = sorted(
            [
                handle_name
                for handle_name in os.listdir(features_dir)
                if "handle" in handle_name
            ]
        )
        score_lst = sorted(
            [
                score_name
                for score_name in os.listdir(features_dir)
                if "score" in score_name
            ]
        )
        score_skin_lst = sorted(
            [
                score_skin_name
                for score_skin_name in os.listdir(features_dir)
                if "skinning" in score_skin_name
            ]
        )

        char_name_lst = [feature_name[:-14] for feature_name in features_lst]
        feat_dict = {}
        for i, feat in enumerate(features_lst):
            feat_path = pjoin(features_dir, feat)
            feat_np = np.load(feat_path)[None,]  # 1 256
            feat_dict[char_name_lst[i]] = feat_np

        handle_dict = {}
        for i, hand in enumerate(handle_lst):
            hand_path = pjoin(features_dir, hand)
            hand_np = np.load(hand_path)[None,]  # 1 40 3
            handle_dict[char_name_lst[i]] = hand_np

        score_dict = {}
        for i, score in enumerate(score_lst):
            score_path = pjoin(features_dir, score)
            score_np = np.load(score_path)[None,]  # 1 N 40
            score_dict[char_name_lst[i]] = score_np

        score_skin_dict = {}
        for i, skin in enumerate(score_skin_lst):
            skin_path = pjoin(features_dir, skin)
            skin_np = np.load(skin_path)[None,]  # 1 N 40
            score_skin_dict[char_name_lst[i]] = skin_np

        joint_dict = {}
        weight_dict = {}
        parents_dict = {}
        handle_weights_dict = {}
        joint_names_dict = {}
        for idx in char_name_lst:
            c = np.load(os.path.join(self.rignet_dir, 'npz', idx + '.npz'))
            center = np.load(
                os.path.join(self.rignet_dir, 'features_30part', idx + '_center.npy')
            )
            scale = np.load(
                os.path.join(self.rignet_dir, 'features_30part', idx + '_scale.npy')
            )
            joints = c["joints"]
            joints = (joints - (center)) / (scale)

            joint_dict[idx] = joints
            parents_dict[idx] = c["parents"]
            weight_dict[idx] = csr_matrix(
                (c["weights_data"], (c["weights_rows"], c["weights_cols"])),
                shape=c["weights_shape"],
                dtype=np.float32,
            ).toarray()

            handle_weight = np.einsum(
                "abc,bd->acd", score_dict[idx], weight_dict[idx]
            )  # 1 40 22
            handle_weights_dict[idx] = handle_weight

        return (
            feat_dict,
            handle_dict,
            char_name_lst,
            joint_dict,
            parents_dict,
            handle_weights_dict,
            weight_dict,
            score_dict,
            score_skin_dict
        )

    def load_mixamo_motion(self):
        self.mixamo_motion = []
        motion_dir = os.path.join(self.mixamo_dir, 'motion')
        self.motion_names = [
            k for k in sorted(os.listdir(motion_dir)) if k.endswith('.npy')
        ]

        self.motion_names = self.motion_names[0:2]                                             # debug

        self.motion_idxs = {}
        self.count = 0
        for name in self.motion_names:
            frames = np.load(os.path.join(motion_dir, name))
            self.motion_idxs[name] = (self.count, self.count + len(frames))
            self.count += len(frames)
            self.mixamo_motion.append(frames)

        self.mixamo_motion_num = len(self.motion_names)
        self.mixamo_motion = np.concatenate(self.mixamo_motion, 0).reshape([-1, 66])

    def load_mixamo_rig(self):
        for idx in self.char_name_lst:
            c = np.load(os.path.join(self.mixamo_dir, 'npz', idx + '.npz'))

            # I don't know why the scale of the mesh is 100 times smaller than the joint. Just for intuition.
            v = c["v"] * 100
            center = (np.max(v, 0, keepdims=True) + np.min(v, 0, keepdims=True)) / 2
            scale = np.max(v[:, 1], 0) - np.min(v[:, 1], 0)
            v = (v - center) / scale
            joints = c["joints"]
            joints = (joints - center) / scale

            self.vs.append(v)
            self.fs.append(c["f"].astype(int))
            self.joints.append(joints)
            self.parents.append(c["parents"])
            self.joint_names.append(c["joint_names"].tolist())
            self.weights.append(
                csr_matrix(
                    (c["weights_data"], (c["weights_rows"], c["weights_cols"])),
                    shape=c["weights_shape"],
                    dtype=np.float32,
                )
            )
            self.tpl_edge_indexs.append(c["tpl_edge_index"].astype(int).T)
            self.geo_edge_indexs.append(
                np.stack(
                    [c["geo_rows"].astype(np.int32), c["geo_cols"].astype(np.int32)], 0
                )
            )

    def extend_motion(self, joint_names, poses):
        # poses T 66
        poses = poses.reshape([poses.shape[0], 22, 3])
        poses[:, 0, :] = 0
        ext_poses = np.zeros(poses.shape, dtype=np.float32)
        for i, name in enumerate(self.MIXAMO_JOINTS):
            assert name in joint_names, "{} not in {}".format(name, joint_names)
            ext_poses[:, joint_names.index(name), :] = poses[:, i, :]
        return ext_poses

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, theta, m_length, text_list = (
            data['motion'],
            data['theta'],
            data['length'],
            data['text'],
        )
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]
        theta = theta[idx : idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # zero padding                               *****************  seq_len 41 9
        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (
                            self.max_motion_length - m_length,
                            motion.shape[1],
                            motion.shape[2],
                        )
                    ),
                ],
                axis=0,
            )
            theta = np.concatenate(
                [
                    theta,
                    np.zeros(
                        (
                            self.max_motion_length - m_length,
                            theta.shape[1],
                            theta.shape[2],
                        )
                    ),
                ],
                axis=0,
            )

        # repeat  padding
        # if m_length < self.max_motion_length:
        #     canvas = np.zeros((self.max_motion_length, motion.shape[1], motion.shape[2]))
        #     repeat_key = int(self.max_motion_length / m_length)
        #     mod = self.max_motion_length % m_length
        #     motion_rp = motion.repeat(repeat_key, axis=0)
        #     canvas[:repeat_key*m_length, :, :] = motion_rp[:, :, :]
        #     canvas[repeat_key*m_length:, :, :] = motion[:mod, :, :]
        #     motion = canvas

        #     canvas_theta = np.zeros((self.max_motion_length, theta.shape[1], theta.shape[2]))
        #     repeat_key = int(self.max_motion_length / m_length)
        #     mod = self.max_motion_length % m_length
        #     theta_rp = theta.repeat(repeat_key, axis=0)
        #     canvas_theta[:repeat_key*m_length, :, :] = theta_rp[:, :, :]
        #     canvas_theta[repeat_key*m_length:, :, :] = theta[:mod, :, :]
        #     theta = canvas_theta

        motion = motion.reshape(motion.shape[0], -1)  # T 41*9
        theta = theta.reshape(theta.shape[0], -1)  # T 22*3

        p = np.random.rand()
        mixamo_prob = 0.4
        focus_v_num = 100
        mixamo_prob = 1.0  # debug
        if p <= mixamo_prob:
            # ramdom choose a mixamo character
            sample_id = random.sample(range(self.mixamo_char_num), 1)
            sample_name = self.char_name_lst[sample_id[0]]
            # sample_name = "Mremireh O Desbiens"
            sample_feature = self.feat_dict[sample_name]
            sample_handle = self.hand_dict[sample_name]
            sample_joint = self.joint_dict[sample_name]
            sample_parents = self.parents_dict[sample_name]
            sample_handle_weights = self.handle_weights_dict[sample_name]
            sample_skin_weight = self.weight_dict[sample_name]
            sample_score_weight = self.score_dict[sample_name]
            sample_score_skin = self.skin_dict[sample_name]

            # random choose a mixamo motion
            motion_id = random.sample(range(self.mixamo_motion_num), 1)
            # motion_id = [0]
            motion_name = self.motion_names[motion_id[0]]
            motion_idx = self.motion_idxs[motion_name]
            motion_sequence = self.mixamo_motion[motion_idx[0] : motion_idx[1]]
            motion_joint_names = self.joint_names_dict[sample_name]

            motion_theta = self.extend_motion(
                motion_joint_names, motion_sequence
            )  # T 22 3

            posed_handle, posed_joints, _ = lbs_batch(
                sample_handle,
                motion_theta,
                sample_joint[None,],
                sample_parents,
                sample_handle_weights,
            )

            mesh_path = os.path.join(self.mixamo_dir, "obj_remesh", sample_name+".obj")
            sample_verts, sample_faces = read_obj(mesh_path)

            center = (np.max(sample_verts, 0, keepdims=True) + np.min(sample_verts, 0, keepdims=True)) / 2
            scale = np.max(sample_verts[:, 1], 0) - np.min(sample_verts[:, 1], 0)
            sample_verts = (sample_verts - center) / scale

            verts_part = np.argmax(sample_score_skin[0], axis=1)     # N
            N = verts_part.shape[0]
            verts_id_lst = [n for n in range(N) if verts_part[n] in self.FOCUS_HANDLES]

            if len(verts_id_lst) >= focus_v_num:
                focus_id_lst = random.sample(verts_id_lst, focus_v_num)
            elif len(verts_id_lst)==0:
                focus_id_lst = random.sample(range(N), focus_v_num)
            else:
                focus_id_lst = verts_id_lst + np.random.choice(range(N), focus_v_num-len(verts_id_lst), replace=True).tolist()
                # focus_id_lst = verts_id_lst + random.sample(verts_id_lst, focus_v_num-len(verts_id_lst))
            focus_verts = sample_verts[focus_id_lst, :]
            focus_weight = sample_skin_weight[focus_id_lst, :]
            focus_skin = sample_score_skin[:, focus_id_lst, :]

            # focus_verts = sample_verts
            # focus_weight = sample_skin_weight

            posed_verts, posed_joints, _ = lbs_batch(
                focus_verts[None,],
                motion_theta,
                sample_joint[None,],
                sample_parents,
                focus_weight[None,],
            )


        else:
            # random choose a rignet char
            sample_id = random.sample(range(self.rignet_char_num), 1)
            sample_name = self.char_name_lst_rig[sample_id[0]]
            # sample_name = "358"
            sample_feature = self.feat_dict_rig[sample_name]
            sample_handle = self.hand_dict_rig[sample_name]
            sample_joint = self.joint_dict_rig[sample_name]
            sample_parents = self.parents_dict_rig[sample_name]
            sample_handle_weights = self.handle_weights_dict_rig[sample_name]
            sample_skin_weight = self.weight_dict_rig[sample_name]
            sample_score_weight = self.score_dict_rig[sample_name]
            sample_score_skin = self.skin_dict_rig[sample_name]

            # random generate a motion
            motion_theta = np.random.normal(
                scale=0.4, size=(196, sample_joint.shape[0], sample_joint.shape[1])
            )
            for t in range(1, 196):
                motion_theta[t, :, :] = (
                    0.9 * motion_theta[t - 1, :, :] + 0.1 * motion_theta[t, :, :]
                )
            motion_theta[:, 0, :] *= 0
            posed_handle, posed_joints, _ = lbs_batch(
                sample_handle,
                motion_theta,
                sample_joint[None,],
                sample_parents,
                sample_handle_weights,
            )

            mesh_path = os.path.join(self.rignet_dir, "obj_remesh", sample_name+".obj")
            sample_verts, sample_faces = read_obj(mesh_path)

            center = (np.max(sample_verts, 0, keepdims=True) + np.min(sample_verts, 0, keepdims=True)) / 2
            scale = np.max(sample_verts[:, 1], 0) - np.min(sample_verts[:, 1], 0)
            sample_verts = (sample_verts - center) / scale

            verts_part = np.argmax(sample_score_skin[0], axis=1)     # N
            N = verts_part.shape[0]
            verts_id_lst = [n for n in range(N) if verts_part[n] in self.FOCUS_HANDLES]
            
            if len(verts_id_lst) >= focus_v_num:
                focus_id_lst = random.sample(verts_id_lst, focus_v_num)
            elif len(verts_id_lst)==0:
                focus_id_lst = random.sample(range(N), focus_v_num)
            else:
                focus_id_lst = verts_id_lst + random.sample(range(N), focus_v_num-len(verts_id_lst))
                # focus_id_lst = verts_id_lst + np.random.choice(verts_id_lst, focus_v_num-len(verts_id_lst), replace=True).tolist()

            focus_verts = sample_verts[focus_id_lst, :]
            focus_weight = sample_skin_weight[focus_id_lst, :]
            focus_skin = sample_score_skin[:, focus_id_lst, :]

            # focus_verts = sample_verts
            # focus_weight = sample_skin_weight

            posed_verts, posed_joints, _ = lbs_batch(
                focus_verts[None,],
                motion_theta,
                sample_joint[None,],
                sample_parents,
                focus_weight[None,],
            )



        if self.mode == "gt":
            return (
                word_embeddings,
                pos_one_hots,
                caption,
                sent_len,
                motion,
                m_length,
                '_'.join(tokens),
            )

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            theta,
            m_length,
            '_'.join(tokens),
            sample_name,
            sample_feature,
            sample_handle,
            posed_handle,
            focus_verts,
            focus_skin,
            posed_verts
        )


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D_T6D_MixRig(data.Dataset):
    def __init__(
        self,
        mode,
        datapath="../datasets/HumanML3D",
        split="train",
        **kwargs,
    ):
        self.mode = mode

        self.dataset_name = 't6d_mixrig'
        self.dataname = 't6d_mixrig'
        abs_base_path = "./"
        mixamo_data_path = (
            "../datasets/Mixamo"
        )
        rignet_data_path = (
            "../datasets/RigNet"
        )
        opt = Namespace()

        opt.dataset_name = self.dataset_name
        opt.dataname = self.dataname
        opt.meta_dir = datapath
        opt.data_root = pjoin(datapath, "HumanML3D")
        opt.motion_dir = pjoin(datapath, "HumanML3D/handle30_6D")
        opt.theta_dir = pjoin(datapath, "HumanML3D/theta")
        opt.text_dir = pjoin(datapath, "HumanML3D/texts")
        opt.mixamo_dir = mixamo_data_path
        opt.rignet_dir = rignet_data_path
        opt.joints_num = 22
        opt.dim_pose = 279
        opt.max_motion_length = 196
        opt.max_text_len = 20
        opt.unit_length = 4
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        self.mean = np.load(pjoin(datapath, 'Mean_30p_6d.npy'))[None,]
        self.std = np.load(pjoin(datapath, 'Std_30p_6d.npy'))[None,]
        self.std[self.std < 1e-5] = 1

        # calculate the mean and std
        # self.mean = np.zeros((1, 31, 9))
        # self.std = np.ones((1, 31, 9))

        if mode == 'eval':
            # self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            # self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
            self.mean_for_eval = np.load(pjoin(datapath, 'Mean_30p_6d.npy'))[None,]
            self.std_for_eval = np.load(pjoin(datapath, 'Std_30p_6d.npy'))[None,]
            self.std[self.std < 1e-5] = 1

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        self.t2m_dataset = Text2MotionDatasetTrans(
            self.opt,
            self.mean,
            self.std,
            self.split_file,
            self.w_vectorizer,
            opt.mixamo_dir,
            opt.rignet_dir,
            mode,
        )
        self.num_actions = 1  # dummy placeholder

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


if __name__ == "__main__":
    pass
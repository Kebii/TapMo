import os
import numpy as np
import torch
import torch.utils.data
from scipy import sparse
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
from utils.geometry import get_tpl_edges, part_scaling, get_normal
from utils.lbs import lbs
from utils.general import np2torch, torch2np
from global_var import *


def normalize_T(T, scale, center):
    M = np.eye(4)
    M[np.arange(3), np.arange(3)] = 1/scale
    M[:3, 3] = -center/scale

    M = torch.from_numpy(M).float().to(T.device)
    M_inv = torch.linalg.inv(M)
    return M@T@M_inv

def joint_augment(joints, scale):
    joints *= scale
    return joints

class HumanML3DDataset(Dataset):
    def __init__(self, motion_path, smpl, simplify=True):
        super(HumanML3DDataset, self).__init__()
        self.no_hand = True
        self.smpl = smpl
        self.simplify = simplify
        self.motion_path_lst = [os.path.join(motion_path, m_name) for m_name in os.listdir(motion_path)]

        self.canonical_v = self.smpl.models['male']['v_template'].astype(np.float32)
        self.canonical_f = self.smpl.models['male']['f']

        self.d_weights = np.array(smpl.models['male']['weights'].astype(np.float32))
        if self.no_hand:
            self.d_weights[:, 20] += np.sum(self.d_weights[:, 22: 37], 1)
            self.d_weights[:, 21] += np.sum(self.d_weights[:, 37: 52], 1)
            self.d_weights = self.d_weights[:, :22]
        self.s_weights = sparse.csr_matrix(self.d_weights)

        if self.simplify:
            assert hasattr(self.smpl, "nearest_face")
            self.low_f = self.smpl.low_f
            self.canonical_v = self.smpl.simplify(self.canonical_v)
            self.canonical_f = self.smpl.low_f
            self.d_weights = self.smpl.simplify(self.d_weights)
            self.s_weights = sparse.csr_matrix(self.d_weights)

        parents = smpl.models['male']['kintree_table'][0].astype(int)
        parents[0] = -1
        if self.no_hand:
            parents = parents[:22]

        self.parents = parents
        tpl_edge_index = get_tpl_edges(self.canonical_v, self.canonical_f)
        self.tpl_edge_index = torch.from_numpy(tpl_edge_index.T).long()
        self.tpl_edge_index, _ = add_self_loops(self.tpl_edge_index, num_nodes=self.canonical_v.shape[0])
        self.tpl_edge_index_np = tpl_edge_index.T

        print("HumanML3D: ", len(self.motion_path_lst))

    def get(self, index):
        if index >= len(self):
            raise StopIteration

        idx1 = index
        tpose_v, pose, shape, gdr, joints, r_velocity, l_velocity, root_y = self.get_data(idx1)

        if self.simplify:
            tpose_v = self.smpl.simplify(tpose_v)

        if self.no_hand:
            joints = joints[:22]

        center = (np.max(tpose_v, 0, keepdims=True) + np.min(tpose_v, 0, keepdims=True)) / 2
        scale = np.max(tpose_v[:, 1], 0) - np.min(tpose_v[:, 1], 0)

        tpose_v = (tpose_v - center) / scale
        joints = (joints - center) / scale

        T = pose.shape[0]

        v1, joints1, T1 = lbs(tpose_v, pose, joints, self.parents, self.d_weights, verbose=True)

        v2, joints2, T2 = lbs(v0_2, theta2, joints0_2, self.parents, self.d_weights, verbose=True)
        aug_v1, aug_joints1, aug_T1 = lbs(aug_v0_1, theta1, aug_joints0_1,  self.parents, self.d_weights, verbose=True)
        aug_v2, aug_joints2, aug_T2 = lbs(aug_v0_2, theta2, aug_joints0_2,  self.parents, self.d_weights, verbose=True)

        v0_1, v0_2, v1, v2, T1, T2 = np2torch(v0_1, v0_2, v1, v2, T1, T2)
        aug_v0_1, aug_v0_2, aug_v1, aug_v2, aug_T1, aug_T2 = np2torch(aug_v0_1, aug_v0_2, aug_v1, aug_v2, aug_T1, aug_T2)

        weights = self.d_weights
        parents = self.parents[None]

        # v_idx = np.where(np.sum(self.d_weights[:, self.head_joint_idxs], 1) > 0.1)[0]
        # f = get_part_mesh(v_idx, self.canonical_f)
        # v0_1 = v0_1[v_idx]
        normal_v0_1 = get_normal(v0_1, self.canonical_f)
        normal_v1 = get_normal(v1, self.canonical_f)
        normal_v0_2 = get_normal(v0_2, self.canonical_f)
        normal_v2 = get_normal(v2, self.canonical_f)


        return Data(v0=v0_1, v1=v1, tpl_edge_index=self.tpl_edge_index, triangle=self.canonical_f[None], name=str(index),
                    aug_v0=aug_v0_1, aug_v1=aug_v1, aug_T=aug_T1, aug_joints=aug_joints0_1[None],
                    feat0=normal_v0_1, feat1=normal_v1,
                    joints=joints0_1[None], weights=weights, parents=parents, theta=theta1[None],
                    num_nodes=len(v0_1), dataset=0), \
               Data(v0=v0_2, v1=v2, tpl_edge_index=self.tpl_edge_index, triangle=self.canonical_f[None], name=str(index),
                    aug_v0=aug_v0_2, aug_v1=aug_v2, aug_T=aug_T2, aug_joints=aug_joints0_2[None],
                    feat0=normal_v0_2, feat1=normal_v2,
                    joints=joints0_2[None], weights=weights, parents=parents, theta=theta2[None],
                    num_nodes=len(v0_2), dataset=0)

    def get_uniform(self, index):
        return self.get(index)

    def get_data(self, idx):
        data_path = self.motion_path_lst[idx]
        data = np.load(data_path)

        r_velocity = data['r_velocity']
        l_velocity = data['l_velocity']
        root_y = data['root_y']
        rot_data_euler = data['rot_data_euler']
        
        shape = np.zeros((10,))
        gdr = -1
        pose = rot_data_euler
        tpose_v, joints = self.smpl(np.zeros([1, 156]), shape[None], [-1], ret_J=True)

        return tpose_v, pose, shape, gdr, joints, r_velocity, l_velocity, root_y


    def len(self):
        return len(self.motion_path_lst)


if __name__ == '__main__':
    from global_var import *
    import cv2
    from models.smpl import SMPL2Mesh
    from utils.render import Renderer
    import matplotlib.pyplot as plt

    smpl = SMPL2Mesh(SMPLH_PATH)
    dataset = AmassDataset(AMASS_PATH, smpl, "train", part_augmentation=True, simplify=True)
    renderer = Renderer(400)

    def equal(a, b):
        print(np.max(np.abs(a-b)))

    def save_obj(path, body_joint, edge):
        with open(path, 'w') as f:
            for i in range(body_joint.shape[0]):
                f.write("v " + str(body_joint[i, 0])+ " " + str(body_joint[i, 1]) + " " + str(body_joint[i, 2]))
                f.write('\n')
            for j in range(edge.shape[0]):
                f.write("f " + str(edge[j, 0]+1) + " " + str(edge[j, 1]+1) + " " + str(edge[j, 2]+1) + "\n")

    for i in range(10):
        idx = np.random.randint(len(dataset))
        data, _ = dataset.get_uniform(idx)
        # print(torch.min(data.v0[:,2]))
        # equal(data.v0.numpy(), data2.v0.numpy())
        # equal(data.v1.numpy(), data.aug_v1.numpy())

        # img = renderer(data.aug_v1.numpy(), data.triangle[0])
        # cv2.imshow('a', np.concatenate((img, img), 1))
        # cv2.waitKey()

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        sequence_containing_x_vals = data.v0[:,0].numpy()
        sequence_containing_y_vals = data.v0[:,1].numpy()
        sequence_containing_z_vals = data.v0[:,2].numpy()

        ax.scatter(sequence_containing_x_vals, sequence_containing_z_vals, sequence_containing_y_vals, s=3)

        plt.savefig('./work_dir/smpl.jpg')

        print(data.triangle.shape)
        print(data.v0.shape)

        save_obj('./work_dir/smpl.obj', data.aug_v0.numpy(), data.triangle[0])

        break
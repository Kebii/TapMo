import os
import numpy as np
import torch
from model.trans2mesh import Trans2Mesh
from os.path import join as pjoin
from kornia.geometry.conversions import quaternion_to_rotation_matrix, QuaternionCoeffOrder
wxyz = QuaternionCoeffOrder.WXYZ

def save_obj(path, body_joint, edge):
   with open(path, 'w') as f:
       for i in range(body_joint.shape[0]):
           f.write("v " + str(body_joint[i, 0])+ " " + str(body_joint[i, 1]) + " " + str(body_joint[i, 2]))
           f.write('\n')
       for j in range(edge.shape[0]):
           f.write("f " + str(edge[j, 0]+1) + " " + str(edge[j, 1]+1) + " " + str(edge[j, 2]+1) + "\n")


if __name__ == "__main__":
    data_path = "./datasets/HumanML3D/joint_trans_all/000000.npz"
    SMPLH_PATH = "./handle_predictor/smplh"
    PART_PATH = "./handle_predictor/smplh"
    save_mesh_path = "./save/data_000000_2"

    if not os.path.exists(save_mesh_path):
        os.makedirs(save_mesh_path)

    trans2mesh = Trans2Mesh(SMPLH_PATH, PART_PATH)

    motion = np.load(data_path)
    body_motion = motion["body_motion"]                     # T 40 7 (4 quaternion + 3 translation)
    root_rotation = motion["root_rotation"]                 # T 4 (quaternion)
    root_veloc = motion["root_veloc"]                       # T 3
    root_pos = motion["root_pos"]                           # T 3
    T = root_rotation.shape[0]
    root_rotation = root_rotation[:, None, :]
    root_veloc = root_veloc[:, None, :]
    root_motion = np.concatenate([root_rotation, root_veloc], axis=2)   # T 1 7
    motion = np.concatenate([body_motion, root_motion], axis=1)         # T 41 7

    sample = torch.from_numpy(motion)[None,].float().permute(0,2,3,1).contiguous()

    bs, N, _, T = sample.shape
    joint_motion = sample[:, :40, :, :] # B N 7 T
    root_motion = sample[:, 40, :, :]       # B 7 T           4 rotation 3 velocity
    root_motion = root_motion.permute(0,2,1).contiguous()

    vertics = trans2mesh(sample)    # B*T V 3

    root_rot_mat = quaternion_to_rotation_matrix(root_motion[:, :, :4].reshape(bs*T, 4).contiguous()).view(bs*T, 3, 3).contiguous()

    vertics = torch.einsum("abc,adc->adb", root_rot_mat, vertics)

    # vertics = torch.einsum("abc,adc->abd", vertics, root_rot_mat)

    position = root_motion[:, :, 4:]

    for t in range(1, T):
        position[:,t,:] = position[:,t,:] + position[:,t-1,:]
    
    vertics = vertics + position.reshape(bs*T, 1, 3)
    vertics = vertics.detach().cpu().numpy()


    face = np.load("./handle_predictor/smplh/face.npy")

    T = vertics.shape[0]

    for t in range(T):
        path = os.path.join(save_mesh_path, 'data_{:05d}.obj'.format(t))
        save_obj(path, vertics[t], face)
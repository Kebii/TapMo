import os
import numpy as np
from tqdm import tqdm
from utils.rotation_convert import rotation_6d_to_matrix, matrix_to_euler_angles
import torch

if __name__ == '__main__':
    source_data_path = "./datasets/HumanML3D/new_joint_vecs_recover"
    save_path = "./datasets/HumanML3D/joint_rot_xyz_recover_tmp"
    os.makedirs(save_path, exist_ok=True)
    source_data_lst = os.listdir(source_data_path)
    for data_name in tqdm(source_data_lst):
        data_path = os.path.join(source_data_path, data_name)
        s_data = np.load(data_path)
        r_velocity, l_velocity, root_y = s_data[:, [0]], s_data[:, [1,2]], s_data[:, [3]]
        rot_data = s_data[:, 67:193]           # T, 21*6
        T = rot_data.shape[0]
        rot_data_matrix = rotation_6d_to_matrix(torch.from_numpy(rot_data.reshape(T, 21, 6)).float())       # T, 21, 3, 3
        rot_data_euler = matrix_to_euler_angles(rot_data_matrix, convention="XYZ")                                            # T, 21, 3
        rot_data_euler = rot_data_euler.numpy()
        rot_data_euler = np.concatenate([np.zeros((T,1,3)), rot_data_euler], axis=1)                                # T, 22, 3

        npz_path = os.path.join(save_path, data_name[:-4])
        np.savez(npz_path,
                 r_velocity=r_velocity,
                 l_velocity=l_velocity,
                 root_y=root_y,
                 rot_data_euler=rot_data_euler)
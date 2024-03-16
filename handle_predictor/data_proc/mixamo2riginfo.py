import numpy as np
from os import listdir, makedirs
from os.path import exists, join

def get_index(lst, item):
    return [(item,index) for (index,value) in enumerate(lst) if value == item]

def save(filename, joint_names, joint_pos, vertices, skinning_weights):
    with open(filename, 'w') as file_info:
        for i, jname in enumerate(joint_names):
            joint_line = 'joints {0} {1:.8f} {2:.8f} {3:.8f}\n'.format(jname, joint_pos[i, 0], joint_pos[i, 1], joint_pos[i, 2])
            file_info.write(joint_line)
        
        file_info.write('root {}\n'.format(joint_names[0]))

        for i in range(vertices.shape[0]):
            cur_line = 'skin {0} '.format(str(i))
            for j, jname in enumerate(joint_names):
                cur_line += '{0} {1:.4f} '.format(jname, float(skinning_weights[i, j]))
            cur_line += '\n'
            file_info.write(cur_line)

        parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        
        this_level = get_index(parents, 0)
        while this_level:
            next_level = []
            for p_node in this_level:
                file_info.write('hier {0} {1}\n'.format(joint_names[p_node[0]], joint_names[p_node[1]]))
                next_level += get_index(parents, p_node[1])
            this_level = next_level

def offset2pos(offset, i, parents, flags):
    if parents[i] == -1:
        return offset[i]
    elif flags[i]:
        offset[i] += offset2pos(offset, parents[i], parents, flags)
        flags[i] = False
        return offset[i]
    else:
        return offset[i]

        
if __name__ == '__main__':
    extract_files_path = "/apdcephfs/private_jiaxuzhang_cq/datasets/skeleton_free/Mixamo/extract"
    save_info_path = "/apdcephfs/private_jiaxuzhang_cq/datasets/skeleton_free/Mixamo/rig_info"
    makedirs(save_info_path, exist_ok=True)
    file_lst = listdir(extract_files_path)
    for file_name in file_lst:
        file_path = join(extract_files_path, file_name)
        info_dic = np.load(file_path)

        save_name = join(save_info_path, file_name.split(".")[0]+'.txt')

        joint_offsets = info_dic['skeleton']
        parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        flag = [True]*len(parents)
        for i in range(len(parents)):
            offset2pos(joint_offsets, i, parents, flag)

        # joint_pos = joint_offsets - joint_offsets[0]
        joint_pos = joint_offsets
        save(save_name, info_dic['joint_names'], joint_pos, info_dic['rest_vertices'], info_dic['skinning_weights'])
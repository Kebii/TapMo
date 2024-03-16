import numpy as np
from utils.mesh_res_mapper import MeshResMapper
from os import listdir, makedirs
from os.path import exists, join
from tqdm import tqdm


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
            verts.append(np.array([float(k) for k in line.split(' ')[1:]]))
        elif line.startswith('f '):
            try:
                onef = np.array([int(k) for k in line.split(' ')[1:]])
            except ValueError:
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


def save_obj(path, body_joint, edge):
   with open(path, 'w') as f:
       for i in range(body_joint.shape[0]):
           f.write("v " + str(body_joint[i, 0])+ " " + str(body_joint[i, 1]) + " " + str(body_joint[i, 2]))
           f.write('\n')
       for j in range(edge.shape[0]):
           f.write("f " + str(edge[j, 0]+1) + " " + str(edge[j, 1]+1) + " " + str(edge[j, 2]+1) + "\n")



if __name__ == '__main__':
    ori_mesh_path = "./demo/man_capo/rest.obj"
    dwn_mesh_path = "./demo/man_capo_remesh/rest.obj"

    tgt_path = "./demo/man_combo"
    save_path = "./demo/man_combo_down"

    ori_v, ori_f = read_obj(ori_mesh_path)
    dwn_v, dwn_f = read_obj(dwn_mesh_path)

    dsampler = MeshResMapper(ori_v, ori_f, dwn_v)

    if not exists(save_path):
            makedirs(save_path)

    file_names = listdir(tgt_path)
    
    for ne in tqdm(file_names):
        pth = join(tgt_path, ne)
        v, f = read_obj(pth)
        v_res = dsampler.upsample(v)
        save_name = join(save_path, ne)
        save_obj(save_name, v_res, dwn_f)



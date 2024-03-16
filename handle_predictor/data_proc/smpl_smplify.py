import numpy as np
# from utils.o3d_wrapper import Mesh
from utils.geometry import get_nearest_face, barycentric
from utils.mesh_res_mapper import MeshResMapper
from global_var import *


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
    high_v, high_f = read_obj("./datasets/temp/smpl.obj")
    low_v, low_f = read_obj("./datasets/temp/simplify_smpl.obj")

    nearest_face = get_nearest_face(low_v, high_v, high_f)
    bary = barycentric(low_v, high_v[high_f[:, 0][nearest_face]],
                       high_v[high_f[:, 1][nearest_face]],
                       high_v[high_f[:, 2][nearest_face]])
    
    sp_v = high_v[high_f[:, 0][nearest_face]] * bary[:, 0:1] + \
              high_v[high_f[:, 1][nearest_face]] * bary[:, 1:2] + \
              high_v[high_f[:, 2][nearest_face]] * bary[:, 2:3]
    
    save_obj("./work_dir/sample_man_sp.obj", sp_v, low_f)

    np.savez(f"{SMPLH_PATH}/simplify.npz", nearest_face=nearest_face, bary=bary, f=low_f)
import os
import numpy as np
import torch
from smplx.lbs import lbs
from tqdm import tqdm


class SMPL2Mesh:
    def __init__(self, bm_path, bm_type='smplh'):
        self.models = {}
        male_npz = np.load(os.path.join(bm_path, 'male/model.npz'))
        female_npz = np.load(os.path.join(bm_path, 'female/model.npz'))
        self.models['male'] = {k: male_npz[k] for k in male_npz}
        self.models['female'] = {k: female_npz[k] for k in female_npz}
        self.f = self.models['male']['f']
        self.bm_type = bm_type

    def __call__(self, pose_all, shape_all, gender_all, ret_J=False):
        if isinstance(pose_all, np.ndarray):
            pose_all = torch.from_numpy(pose_all).to(torch.float32)
        if isinstance(shape_all, np.ndarray):
            shape_all = torch.from_numpy(shape_all).to(torch.float32)
        if isinstance(gender_all, torch.Tensor):
            gender_all = gender_all.detach().cpu().numpy()

        all_size = len(gender_all)
        batch_size = 256
        batch_num = 1 + (all_size - 1) // batch_size
        mesh_all = []
        J_all = []
        for bi in range(batch_num):
            l = bi * batch_size
            r = np.minimum((bi + 1) * batch_size, all_size)
            cur_bs = r - l
            pose = pose_all[l: r]
            shape = shape_all[l: r]
            gender = gender_all[l: r]

            gender_ind = {}
            gender_ind['male'] = [idx for (idx, g) in enumerate(gender) if g==-1]
            gender_ind['female'] = [idx for (idx, g) in enumerate(gender) if g==1]

            verts = {}
            Js = {}
            for gdr in ['male', 'female']:
                if not gender_ind[gdr]:
                    continue

                gdr_betas = shape[gender_ind[gdr]]
                gdr_pose = pose[gender_ind[gdr]]

                v_template = np.repeat(self.models[gdr]['v_template'][np.newaxis], len(gdr_betas), axis=0)
                v_template = torch.tensor(v_template, dtype=torch.float32)


                shapedirs = torch.tensor(self.models[gdr]['shapedirs'], dtype=torch.float32)

                posedirs = self.models[gdr]['posedirs']
                posedirs = posedirs.reshape(posedirs.shape[0]*3, -1).T
                posedirs = torch.tensor(posedirs, dtype=torch.float32)
                # if no_psd:
                #     posedirs = 0 * posedirs

                J_regressor = torch.tensor(self.models[gdr]['J_regressor'], dtype=torch.float32)

                parents = torch.tensor(self.models[gdr]['kintree_table'][0], dtype=torch.int32).long()

                lbs_weights = torch.tensor(self.models[gdr]['weights'], dtype=torch.float32)

                v, J = lbs(gdr_betas, gdr_pose, v_template, shapedirs, posedirs,
                           J_regressor, parents, lbs_weights)

                verts[gdr] = v
                Js[gdr] = J

            mesh = torch.zeros(cur_bs, 6890, 3)
            Js_batch = torch.zeros(cur_bs, 52, 3)

            for gdr in ['male', 'female']:
                if gdr in verts:
                    mesh[gender_ind[gdr]] = verts[gdr]
                    Js_batch[gender_ind[gdr]] = Js[gdr]

            mesh_all.append(mesh)
            if ret_J:
                J_all.append(Js_batch)
        mesh_all = torch.cat(mesh_all, 0)

        if ret_J:
            J_all = torch.cat(J_all, 0)
            return mesh_all, J_all
        else:
            return mesh_all
        
    
def load_smpl_mesh(smpl, betas):
    # shape = np.zeros((16,))
    gdr = -1
    v_tpose, joints = smpl(np.zeros([1, 156]), betas[None], [-1], ret_J=True)
    v_tpose = v_tpose.numpy()
    joints = joints.numpy()

    return v_tpose[0], joints[0], betas, gdr

def save_obj(path, body_joint, edge):
    with open(path, 'w') as f:
        for i in range(body_joint.shape[0]):
            f.write("v " + str(body_joint[i, 0])+ " " + str(body_joint[i, 1]) + " " + str(body_joint[i, 2]))
            f.write('\n')
        for j in range(edge.shape[0]):
            f.write("f " + str(edge[j, 0]+1) + " " + str(edge[j, 1]+1) + " " + str(edge[j, 2]+1) + "\n")


def generate_smpl_meshs():
    smpl_path = "./smplh"
    save_path = "./datasets/skeleton_free/SMPL_meshs"
    beta_range = [i for i in range(-10,11,2)]
    beta_num = 5
    smpl_model = SMPL2Mesh(smpl_path)

    for i in tqdm(range(beta_num)):
        for j in beta_range:
            shape = np.zeros((16,))
            # shape[i] = float(j)
            shape[0] = -8.0
            shape[1] = -6.0
            vt, joints, shape, gdr = load_smpl_mesh(smpl_model, shape)

            center = (np.max(vt, 0, keepdims=True) + np.min(vt, 0, keepdims=True)) / 2
            scale = np.max(vt[:, 1], 0) - np.min(vt[:, 1], 0)

            vt = (vt - center) / scale
            joints = (joints - center) / scale
            joints = joints[:22, ]

            # npz_name = "smpl_{}_{}.npz".format(str(i), str(j))
            # obj_name = "smpl_{}_{}.obj".format(str(i), str(j))
            npz_name = "smpl_0_-8_1_-6.npz"
            obj_name = "smpl_0_-8_1_-6.obj"
            npz_save_name = os.path.join(save_path, npz_name)
            obj_save_name = os.path.join(save_path, obj_name)

            np.savez(npz_save_name,
             verts = vt,
             joints = joints,
             betas = shape
             )
            
            save_obj(obj_save_name, vt, smpl_model.f)
            break
        break

if __name__ == '__main__':
    generate_smpl_meshs()



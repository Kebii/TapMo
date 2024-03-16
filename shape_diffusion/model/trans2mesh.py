import os
import numpy as np
import torch
from kornia.geometry.conversions import quaternion_to_rotation_matrix, QuaternionCoeffOrder
from kornia.geometry.conversions import angle_axis_to_rotation_matrix
wxyz = QuaternionCoeffOrder.WXYZ
from smplx.lbs import lbs
import random
from smplx.lbs import batch_rigid_transform
from scipy.spatial.transform import Rotation as rot


def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat

def handle2mesh(transformation, handle_pos, region_score, v0):
    """
    use per-part trans+rot to reconstruct the mesh
    transformation: (B, 41, 3+4)
    handle_pos: handle position of T-pose
    handle_pos: (B, 40, 3)
    region_score: (B, V, 40)
    v0: (V, 3)
    """
    B, K, _ = handle_pos.shape   # bs 40 3
    V = v0.shape[0]
    bs, d, _, T = transformation.shape
    transformation = transformation.reshape(bs, K+1, 7, T).permute(0,3,1,2).contiguous().reshape(bs*T, K+1, 7)[:, :K, :]
    disp = transformation[:, :, :3]         # B K 3
    rot = transformation[:, :, 3:]

    # why order=wxyz can not be change
    rot = quaternion_to_rotation_matrix(rot.reshape(bs*T*K, 4).contiguous(), order=wxyz).view(bs*T, K, 3, 3).contiguous()

    hd_disp = disp.unsqueeze(1).repeat(1, V, 1, 1)      # B V K 3
    hd_rot = rot.unsqueeze(1).repeat(1, V, 1, 1, 1)   # B V K 3 3
    hd_pos = handle_pos.unsqueeze(1).repeat(1, V, 1, 1)      # B V K 3
    # print(hd_rot.shape, v0.shape, hd_pos.shape, hd_disp.shape)
    per_hd_v = torch.einsum("abcde,abce->abcd", hd_rot, (v0.unsqueeze(0).unsqueeze(2) - hd_pos)) + hd_pos + hd_disp  # (B, V, K, 3)
    # print(per_hd_v.shape, region_score.shape)
    v = torch.sum(region_score[:, :, :, None] * per_hd_v, 2)  # (B, V, 3)
    return v


def trans2hdpos(transformation, handle_pos):
        device = transformation.device
        B, K, _ = handle_pos.shape   # bs 40 3
        bs, d, _, T = transformation.shape
        transformation = transformation.reshape(bs, K+1, 9, T).permute(0,3,1,2).contiguous().reshape(bs*T, K+1, 9)[:, :K, :]
        disp = transformation[:, :, :3]         # B K 3
        rot = transformation[:, :, 3:]
        # rot = quaternion_to_rotation_matrix(rot.reshape(bs*T*K, 4).contiguous(), order=wxyz).view(bs*T, K, 3, 3).contiguous()
        rot = repr6d2mat(rot)

        hd_disp = disp      # B K 3
        hd_rot = rot        # B K 3 3
        hd_pos = handle_pos.to(device).unsqueeze(1).repeat(bs, T, 1, 1).reshape(bs*T, K, 3)      # B*T K 3

        per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, hd_pos) + hd_disp  # (B, K, 3)
     
        return per_hd_v.reshape(bs, T, K, 3).permute(0, 2, 3, 1).contiguous().reshape(bs, -1, 1, T), hd_pos.reshape(bs, T, K, 3).permute(0, 2, 3, 1).contiguous().reshape(bs, -1, 1, T)

def trans2hdpos_(transformation, handle_pos):
        device = transformation.device
        B, K, _ = handle_pos.shape   # bs 40 3
        bs, d, _, T = transformation.shape
        transformation = transformation.reshape(bs, K+1, 9, T).permute(0,3,1,2).contiguous().reshape(bs*T, K+1, 9)[:, :K, :]
        disp = transformation[:, :, :3]         # B K 3

        hd_disp = disp      # B K 3
        hd_pos = handle_pos.to(device).unsqueeze(1).repeat(bs, T, 1, 1).reshape(bs*T, K, 3)      # B*T K 3

        per_hd_v = hd_pos + hd_disp  # (B, K, 3)
     
        return per_hd_v.reshape(bs, T, K, 3).permute(0, 2, 3, 1).contiguous().reshape(bs, -1, 1, T), hd_pos.reshape(bs, T, K, 3).permute(0, 2, 3, 1).contiguous().reshape(bs, -1, 1, T)

def lbs_batch(v, rotations, J, parents, weights):
    """
    Args:
        v: (B, V, 3)
        rotations: (B, J, 3), rotation vector numpy
        J: (B, J, 3), joint positions
        parents: (J), kinematic chain indicator
        weights: (B, V, J), skinning weights

    Returns:
        articulated vertices: (B, V, 3)
    """
    B, num_joints, _ = J.shape
    rot_mats = angle_axis_to_rotation_matrix(rotations.reshape([-1, 3])).reshape([B, num_joints, 3, 3])
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    # W = np.tile(weights[None], [B, 1, 1])
    # W = torch.tile(weights, [B, 1, 1])
    W = weights
    T = torch.einsum("bnj,bjpq->bnpq", W, A)
    R = T[:, :, :3, :3]
    t = T[:, :, :3, 3]
    verts = torch.einsum("bnpq,bnq->bnp", R, v) + t
    return verts

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


def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.append(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index

class Trans2Mesh:
    def __init__(self, smpl_path, part_path):
        self.smpl = SMPL2Mesh(smpl_path)
        vt, joints, self.shape, self.gdr = self.load_smpl_mesh(self.smpl)
        self.center = (np.max(vt, 0, keepdims=True) + np.min(vt, 0, keepdims=True)) / 2
        self.scale = np.max(vt[:, 1], 0) - np.min(vt[:, 1], 0)

        self.vt_scale = torch.from_numpy(vt).float()
        self.vt = torch.from_numpy((vt-self.center)/self.scale).float()
        self.joints_scale = torch.from_numpy(joints).float()
        self.joints = torch.from_numpy((joints-self.center)/self.scale).float()
        self.joints = self.joints[:22, ]
        self.f = self.smpl.models['male']['f']

        parents = self.smpl.models['male']['kintree_table'][0].astype(int)
        parents[0] = -1
        self.parents = torch.from_numpy(parents[:22]).long()

        d_weights = np.array(self.smpl.models['male']['weights'].astype(np.float32))
        d_weights[:, 20] += np.sum(d_weights[:, 22: 37], 1)
        d_weights[:, 21] += np.sum(d_weights[:, 37: 52], 1)
        self.weights = torch.from_numpy(d_weights[:, :22]).float()

        handle_pos = np.load(os.path.join(part_path, "handle30_pos.npy"))
        region_score = np.load(os.path.join(part_path, "region30_score.npy"))

        self.handle_pos = torch.from_numpy(handle_pos).float()            # (1, 40, 3)
        self.region_score = torch.from_numpy(region_score).float()        # (1, V, 40)

        tpl_edge_index = get_tpl_edges(vt, self.f)
        tpl_edge_index = torch.from_numpy(tpl_edge_index.T).long()             # 2 N
        # tpl_edge_index, _ = add_self_loops(self.tpl_edge_index, num_nodes=self.canonical_v.shape[0])
        self_loop = torch.tensor([[i, i] for i in range(vt.shape[0])]).permute(1,0)
        self.tpl_edge_index = torch.cat([tpl_edge_index, self_loop], dim=1)


    def load_smpl_mesh(self, smpl):
        shape = np.zeros((16,))
        gdr = -1
        v_tpose, joints = smpl(np.zeros([1, 156]), shape[None], [-1], ret_J=True)
        v_tpose = v_tpose.numpy()
        joints = joints.numpy()

        return v_tpose[0], joints[0], shape, gdr
    
    def trans2hdpos(self, x):
        return trans2hdpos(x, self.handle_pos)
    
    def trans2hdpos_(self, x):
        return trans2hdpos_(x, self.handle_pos)
    
    def theta_lbs(self, theta):
        device = theta.device
        # theta = theta.detach().cpu().numpy()
        B = theta.shape[0]
        verts = lbs_batch(self.vt[None,].repeat(B,1,1), theta, self.joints[None,].repeat(B,1,1), self.parents, self.weights[None,].repeat(B,1,1))
        verts =  verts * self.scale + torch.from_numpy(self.center).to(device)
        return verts
    
    def __call__(self, transformation):
        """
        use per-part trans+rot to reconstruct the mesh
        transformation: (B, 41, 3+4, T)
        handle_pos: handle position of T-pose
        self.handle_pos: (1, 40, 3)
        self.region_score: (V, 40)
        v0: (V, 3)
        """
        B = transformation.shape[0]
        T = transformation.shape[3]
        device = transformation.device

        handle_pos = self.handle_pos.repeat(B*T, 1, 1).to(device)
        region_score = self.region_score.repeat(B*T, 1, 1).to(device)

        vt = self.vt.to(device)

        vp = handle2mesh(transformation, handle_pos, region_score, vt)

        vp = vp * self.scale + torch.from_numpy(self.center).to(device)

        return vp



if __name__ == "__main__":

    SMPLH_PATH = "./smplh"
    PART_PATH = "./smplh"

    trans2mesh = Trans2Mesh(SMPLH_PATH, PART_PATH)
import os
import numpy as np
import torch
import torch.utils.data
from scipy import sparse
from tqdm import tqdm
import argparse

from torch_geometric.utils import add_self_loops, scatter
from utils.geometry import get_tpl_edges, part_scaling, get_normal
from utils.lbs import lbs
from utils.general import np2torch, torch2np
from models.smpl import SMPL2Mesh
from models.networks import HandlePredictorSWTpl
from models.ops import handle2mesh, get_transformation, arap_smooth2, OneEuroFilter
from utils.o3d_wrapper import Mesh
import trimesh

import cv2
from scipy.spatial.transform import Rotation as R

def load_smpl_mesh(smpl):

    shape = np.zeros((16,))
    gdr = -1
    v_tpose, joints = smpl(np.zeros([1, 156]), shape[None], [-1], ret_J=True)
    v_tpose = v_tpose.numpy()
    joints = joints.numpy()

    return v_tpose[0], joints[0], shape, gdr



def load_target_mesh(tgt_path):
    m = Mesh(filename=tgt_path)
    tpl_edge_index = get_tpl_edges(m.v, m.f)
    tpl_edge_index = tpl_edge_index.astype(int).T
    tpl_edge_index = torch.from_numpy(tpl_edge_index).long()
    tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=m.v.shape[0])

    center = (np.max(m.v, 0, keepdims=True) + np.min(m.v, 0, keepdims=True)) / 2
    scale = np.max(m.v[:, 1], 0) - np.min(m.v[:, 1], 0)
    # scale = np.max(m.v, 0, keepdims=True) - np.min(m.v, 0, keepdims=True)
    v_t = (m.v - center) / scale
    return v_t, m.f, tpl_edge_index, center, scale


def load_skw_predictor(ckpt_path):
    input_dim = 6
    predictor = HandlePredictorSWTpl(input_dim, 30)
    checkpoint = torch.load(ckpt_path)
    predictor.load_state_dict(checkpoint['predictor'])
    predictor.eval()

    return predictor

def load_trans(motion_path):
    data = np.load(motion_path)
    return data["body_motion"], data["root_rotation"], data["root_pos"]


def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    res = [x, y, z]
    res = [np.expand_dims(v, axis=-2) for v in res]
    mat = np.concatenate(res, axis=-2)
    return mat

def skin_pooling(skin, tpl, k=1):
    row, col = tpl
    skin_p = skin
    for i in range(k):
        skin_p = scatter(skin_p[row], col, dim=0, reduce='mean')
    return skin_p

def trans2mesh(motion_path, tgt_path, ckpt_path, save_dir, smpl, simplify=False):
    no_hand = True
   
    canonical_v = smpl.models['male']['v_template'].astype(np.float32)
    canonical_f = smpl.models['male']['f']

    d_weights = np.array(smpl.models['male']['weights'].astype(np.float32))
    if no_hand:
        d_weights[:, 20] += np.sum(d_weights[:, 22: 37], 1)
        d_weights[:, 21] += np.sum(d_weights[:, 37: 52], 1)
        d_weights = d_weights[:, :22]

    if simplify:
        assert hasattr(smpl, "nearest_face")
        low_f = smpl.low_f
        canonical_v = smpl.simplify(canonical_v)
        canonical_f = smpl.low_f.astype(np.int32)
        d_weights = smpl.simplify(d_weights)

    parents = smpl.models['male']['kintree_table'][0].astype(int)
    parents[0] = -1
    if no_hand:
        parents = parents[:22]

    tpl_edge_index = get_tpl_edges(canonical_v, canonical_f)
    tpl_edge_index = torch.from_numpy(tpl_edge_index.T).long()
    tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=canonical_v.shape[0])

    print("HumanML3D: ", motion_path)

    # load predictor
    predictor = load_skw_predictor(ckpt_path)

    tgt_v, tgt_f, tgt_tpl, center_t, scale_t = load_target_mesh(tgt_path)
    tpl_t = tgt_tpl
    f_t = torch.from_numpy(tgt_f).long()
    v_t = torch.from_numpy(tgt_v).float()
    normal_v_t = get_normal(v_t, tgt_f)

    v_tpose, joints, shape, gdr = load_smpl_mesh(smpl)

    if simplify:
        v_tpose = smpl.simplify(v_tpose)

    if no_hand:
        joints = joints[:22]

    center = joints[0]
    scale = np.max(v_tpose[:, 1], 0) - np.min(v_tpose[:, 1], 0)

    v_ts = (v_t*scale + center).numpy()
    v_tpose = (v_tpose - center) / scale
    joints = (joints - center) / scale

    v0 = np2torch(v_tpose)[0]
    normal_v0 = get_normal(v0, canonical_f)

    body_motion, root_rotation, root_position = load_trans(motion_path)
    
    T = body_motion.shape[0]

    batch_t = torch.zeros(v_t.shape[0]).to(torch.int64) 
    hm_t, hd_t_mean, __, region_score_t = predictor(torch.cat((v_t, normal_v_t), 1), tpl_edge_index=tpl_t, batch=batch_t, verbose=True)
    skin_p = skin_pooling(region_score_t, tpl_t, k=2)

    # ini one euro filter
    frames = 120
    start = 0
    end = 4 * np.pi
    filter_t = np.linspace(start, end, frames)
    min_cutoff = 1.7
    beta = 0.3
    d_cutoff = 20


    for t in tqdm(range(T)):
        body_motion_smooth = arap_smooth2(torch.from_numpy(body_motion[t][None]).float(), hd_t_mean, skin_p, batch_t, v_t, tpl_t, 0.01)
        if t == 0:
            one_euro_filter = OneEuroFilter(
                                            filter_t[0], body_motion_smooth,
                                            min_cutoff=min_cutoff,
                                            beta=beta,
                                            d_cutoff=d_cutoff
                                            )
        else:
            body_motion_smooth = one_euro_filter(filter_t[t], body_motion_smooth)


        root_rot_mat = repr6d2mat(root_rotation[t])
        pred_v = handle2mesh(body_motion_smooth, hd_t_mean, skin_p, batch_t, v_t)
        pred_v = pred_v.detach().numpy()
        pred_v = np.dot(root_rot_mat, pred_v.transpose(1,0)).transpose(1,0)
        pred_v = pred_v * scale + center
        pred_v = pred_v + root_position[t][None]

        trimesh_obj = trimesh.Trimesh(vertices=pred_v, faces=tgt_f)
        smooth_mesh = trimesh.smoothing.filter_taubin(mesh=trimesh_obj, lamb=0.2, nu=0.2, iterations=10)
        Mesh(v=smooth_mesh.vertices, f=smooth_mesh.faces).write_obj(os.path.join(save_dir, 'pose_{:05d}.obj'.format(t)))


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='Motion to mesh sequence')
    parser.add_argument(
        '--motion_path',
        default='../shape_diffusion/save/motion_0.npz',
        help='path to the generated motion file',
    )
    parser.add_argument(
        '--tgt_mesh_path',default='../demo/mesh/ghost.obj', help='path to the target mesh'
    )
    parser.add_argument(
        '--ckpt_path', default='../weights/handle_predictor_latest.pth', help='path to the model weights'
    )
    
    parser.add_argument(
        '--save_dir', default='../demo/results/001', help='directory to save the meshes'
    )

    return parser

def main(motion_path, tgt_mesh_path, ckpt_path, save_dir):
    SMPLH_PATH = "./smplh"
    smpl = SMPL2Mesh(SMPLH_PATH)
    os.makedirs(save_dir, exist_ok=True)
    trans2mesh(motion_path, tgt_mesh_path, ckpt_path, save_dir, smpl)
    
if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()
    main(arg.motion_path, arg.tgt_mesh_path, arg.ckpt_path, arg.save_dir)

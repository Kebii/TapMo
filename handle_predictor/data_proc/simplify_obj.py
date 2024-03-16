import os
import pymeshlab
from global_var import *
from utils.o3d_wrapper import Mesh
from tqdm import tqdm




ms = pymeshlab.MeshSet()
obj_path = "/apdcephfs/private_jiaxuzhang_cq/datasets/temp/smpl.obj"
save_path = "/apdcephfs/private_jiaxuzhang_cq/datasets/temp/simplify_smpl.obj"
ms.load_new_mesh(obj_path)
face_num = len(Mesh(filename=obj_path).f)
if face_num > 10000:
    face_num = 5000
elif face_num > 3000:
    face_num = (face_num - 3000) * (2/7) + 3000
    face_num = int(face_num)
else:
    face_num = -1

ms.remove_isolated_pieces_wrt_face_num(mincomponentsize=25, removeunref=True)
if face_num > 0:
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=face_num, autoclean=True)
ms.save_current_mesh(save_path, save_vertex_normal=False)
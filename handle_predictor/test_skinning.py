import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import DataLoader
from kornia.geometry.conversions import QuaternionCoeffOrder
wxyz = QuaternionCoeffOrder.WXYZ
from models.networks import PerPartEncoderTpl, PerPartDecoder, HandlePredictorSWTpl
from models.ops import handle2mesh, get_transformation, arap_smooth, get_transformation_root
from utils.o3d_wrapper import Mesh
from data_utils.custom_loader import CustomDataset, CustomMotionDataset
from utils.visualization import visualize_part, visualize_handle


def save_obj(path, body_joint, joint_color, edge):
   with open(path, 'w') as f:
       for i in range(body_joint.shape[0]):
           f.write("v " + str(body_joint[i, 0])+ " " + str(body_joint[i, 1]) + " " + str(body_joint[i, 2]) + " " + str(joint_color[i, 0])+ " " + str(joint_color[i, 1]) + " " + str(joint_color[i, 2]))
           f.write('\n')
       for j in range(edge.shape[0]):
           f.write("f " + str(edge[j, 0]+1) + " " + str(edge[j, 1]+1) + " " + str(edge[j, 2]+1) + "\n")

if __name__ == '__main__':
    use_smooth = True
    save_dir = '/apdcephfs/private_jiaxuzhang_cq/datasets/skeleton_free/evaluate/eval_failure_skin/'
    ckpt_path = "/apdcephfs/private_jiaxuzhang_cq/code/skfree_workdir/train_3_6_rootsw_30part/exp/latest.pth"
    mesh_data_dir = "/apdcephfs/private_jiaxuzhang_cq/datasets/skeleton_free/evaluate/eval_failure_mesh"

    os.makedirs(save_dir, exist_ok=True)

    input_dim = 6
    predictor = HandlePredictorSWTpl(input_dim, 30)

    device = torch.device("cuda:0")
    predictor.to(device)


    checkpoint = torch.load(ckpt_path)
    predictor.load_state_dict(checkpoint['predictor'])

    predictor.eval()

    torch.set_grad_enabled(False)

    dst_set = CustomDataset(mesh_data_dir)
    dst_loader = DataLoader(dst_set, batch_size=1,
                            shuffle=False, pin_memory=False, drop_last=False)


    for i_d, dst_data in enumerate(tqdm(dst_loader)):
        dst_data.to(device)
        hm0, hd0_mean, _, region_score0 = predictor(torch.cat((dst_data.v0, dst_data.feat0), 1)
                                                    , data=dst_data, verbose=True)
        dst_skw = region_score0.detach().cpu().numpy()
        skw = dst_skw
        vv, ff, vcc = visualize_part(dst_data.v0, dst_data.triangle[0][0], None, skw, save_path=None)
        # vv, ff, vcc = visualize_handle(dst_data.v0, dst_data.triangle[0][0], hd0_mean[0], save_path=None)
        save_obj(os.path.join(save_dir, dst_data.name[0]+".obj"), vv[0], vcc[0], ff[0])


    

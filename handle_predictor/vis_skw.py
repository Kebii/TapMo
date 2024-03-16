from utils.visualization import visualize_part
from utils.o3d_wrapper import Mesh
import numpy as np

def save_obj(path, body_joint, joint_color, edge):
   with open(path, 'w') as f:
       for i in range(body_joint.shape[0]):
           f.write("v " + str(body_joint[i, 0])+ " " + str(body_joint[i, 1]) + " " + str(body_joint[i, 2]) + " " + str(joint_color[i, 0])+ " " + str(joint_color[i, 1]) + " " + str(joint_color[i, 2]))
           f.write('\n')
       for j in range(edge.shape[0]):
           f.write("f " + str(edge[j, 0]+1) + " " + str(edge[j, 1]+1) + " " + str(edge[j, 2]+1) + "\n")



if __name__ == '__main__':
    mesh_path = "./skfree_demo/Ortiz_T_remesh/Ortiz_T.obj"
    skw_path = "./skfree_demo/my_results/train_12_28_centersw/Ortiz_skin_1_2_rootsw_jointaug03_150.npy"
    save_path = "./skfree_demo/my_results/train_1_2_rootsw_jointaug03_150/Ortiz_skin_1_2_rootsw_jointaug03_150.obj"
    m = Mesh(filename=mesh_path)
    skw = np.load(skw_path)
    print(skw.shape)
    vv, ff, vcc = visualize_part(m.v, m.f, None, skw, save_path=None)
    save_obj(save_path, vv[0], vcc[0], ff[0])
    
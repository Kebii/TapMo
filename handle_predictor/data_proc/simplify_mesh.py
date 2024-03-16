import pymeshlab as ml


mesh_path = "/apdcephfs/private_jiaxuzhang_cq/code/motion-diffusion-model/train/save/mesh/tortoise_edge_1_k3_arapt05_sm/pose_00000.obj"
save_path = "/apdcephfs/private_jiaxuzhang_cq/code/motion-diffusion-model/train/save/mesh/test_simplify.obj"

ms = ml.MeshSet()
ms.load_new_mesh(mesh_path)
m = ms.current_mesh()
print('input mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

#Target number of vertex
TARGET=300

#Estimate number of faces to have 100+10000 vertex using Euler
numFaces = 100 + 2*TARGET

#Simplify the mesh. Only first simplification will be agressive
while (ms.current_mesh().vertex_number() > TARGET):
    ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
    print("Decimated to", numFaces, "faces mesh has", ms.current_mesh().vertex_number(), "vertex")
    #Refine our estimation to slowly converge to TARGET vertex number
    numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

m = ms.current_mesh()
print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
ms.save_current_mesh(save_path)
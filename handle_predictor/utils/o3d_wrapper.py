import numpy as np
import open3d as o3d
USE_PYMESH = True
try:
    import pymesh
except ImportError as e:
    print(e)
    print("Failed to load pymesh. Open3d will be used")
    USE_PYMESH = False


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

# preprocess RigNet dataset
def read_obj2(path):
    """
    read verts and faces from obj file. This func will convert quad mesh to triangle mesh
    """
    with open(path) as f:
        lines = f.read().splitlines()
    verts = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            part = line.split(' ')[1:]
            if len(part)!=3:
                continue
            vert = np.array([float(k) if k!='' else 0.0 for k in part])
            verts.append(vert)
        elif line.startswith('f '):
            try:
                if "//" in line:
                    onef = np.array([int(k.split("//")[0]) for k in line.split(' ')[1:]])
                else:
                    onef = np.array([int(k.split("/")[0]) for k in line.split(' ')[1:]])
            except ValueError:
                print(line)
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


def read_obj_full(path):
    with open(path) as f:
        lines = f.read().splitlines()
    verts = []
    v_faces = []
    vts = []
    vt_faces = []
    for line in lines:
        if line.startswith('v '):
            part = line.split(' ')[1:]
            if len(part)!=3:
                continue
            vert = np.array([float(k) if k!='' else 0.0 for k in part])
            verts.append(vert)
        elif line.startswith('vt '):
            part = line.split(' ')[1:]
            if len(part)!=2:
                continue
            vt = np.array([float(k) if k!='' else 0.0 for k in part])
            vts.append(vt)
        elif line.startswith('f '):
            try:
                if "//" in line:
                    v_onef = np.array([int(k.split("//")[0]) for k in line.split(' ')[1:]])
                    vt_onef = np.array([int(k.split("//")[1]) for k in line.split(' ')[1:]])
                else:
                    v_onef = np.array([int(k.split("/")[0]) for k in line.split(' ')[1:]])
                    vt_onef = np.array([int(k.split("/")[1]) for k in line.split(' ')[1:]])
            except ValueError:
                print(line)
                continue
            if len(v_onef) == 4:
                v_faces.append(v_onef[[0, 1, 2]])
                v_faces.append(v_onef[[0, 2, 3]])
            elif len(v_onef) > 4:
                pass
            else:
                v_faces.append(v_onef)
            if len(vt_onef) == 4:
                vt_faces.append(vt_onef[[0, 1, 2]])
                vt_faces.append(vt_onef[[0, 2, 3]])
            elif len(vt_onef) > 4:
                pass
            else:
                vt_faces.append(vt_onef)

    return np.stack(verts), np.stack(vts), np.stack(v_faces)-1, np.stack(vt_faces)-1

def save_obj_full(path, v, v_faces, vt, vt_faces, mtl_name):
    with open(path, 'w') as f:
        f.write("mtllib {}.mtl".format(mtl_name) + "\n")
        f.write("usemtl {}".format(mtl_name) + "\n")
        v_num = v.shape[0]
        vt_num = vt.shape[0]
        f_num = v_faces.shape[0]
        for i in range(v_num):
            f.write("v " + str(v[i, 0])+ " " + str(v[i, 1]) + " " + str(v[i, 2]))
            f.write('\n')
        for i in range(vt_num):
            f.write("vt " + str(vt[i, 0])+ " " + str(vt[i, 1]))
            f.write('\n')
        for i in range(f_num):
            f.write("f " + "{}/{}".format(str(v_faces[i,0]+1), str(vt_faces[i,0]+1)) + " "
                    + "{}/{}".format(str(v_faces[i,1]+1), str(vt_faces[i,1]+1)) + " "
                    "{}/{}".format(str(v_faces[i,2]+1), str(vt_faces[i,2]+1)))
            f.write('\n')



class MeshO3d(object):
    def __init__(self, v=None, f=None, filename=None):
        self.m = o3d.geometry.TriangleMesh()
        if v is not None:
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        elif filename is not None:
            v, f = read_obj2(filename)
            self.m = o3d.geometry.TriangleMesh()
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))

    @property
    def v(self):
        return np.asarray(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = o3d.utility.Vector3dVector(value.astype(np.float32))

    @property
    def f(self):
        return np.asarray(self.m.triangles)

    @f.setter
    def f(self, value):
        self.m.triangles = o3d.utility.Vector3iVector(value.astype(np.int32))

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=True)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=False, compressed=True)


class MeshPyMesh(object):
    def __init__(self, v=None, f=None, vc=None, filename=None):
        if v is not None:
            self.m = pymesh.form_mesh(v, f)
        elif filename is not None:
            self.m = pymesh.load_mesh(filename)
        self.m.add_attribute('vertex_red')
        self.m.add_attribute('vertex_green')
        self.m.add_attribute('vertex_blue')
        if vc is not None:
            self.vc = vc

    @property
    def v(self):
        return np.copy(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = value

    @property
    def f(self):
        return np.copy(self.m.faces)

    @f.setter
    def f(self, value):
        self.m.faces = value

    @property
    def vc(self):
        return np.stack((self.m.get_attribute('vertex_red'), self.m.get_attribute('vertex_green'),
                         self.m.get_attribute('vertex_blue')), 1)/255

    @vc.setter
    def vc(self, value):
        value = np.copy(value) * 255
        self.m.set_attribute('vertex_red', value[:, 0])
        self.m.set_attribute('vertex_green', value[:, 1])
        self.m.set_attribute('vertex_blue', value[:, 2])

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        pymesh.save_mesh(fpath, self.m)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        if self.vc.size == 0:
            pymesh.save_mesh(fpath, self.m)
        else:
            # import IPython; IPython.embed()
            pymesh.save_mesh(fpath, self.m, 'vertex_red', 'vertex_green', 'vertex_blue')

if USE_PYMESH:
    Mesh = MeshPyMesh
else:
    Mesh = MeshO3d
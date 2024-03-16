import os

# Please update the below to your own paths
# mirrors.tencent.com/jiaxuzhang/skelfree_python3_miniconda3_torch1.13.0_pyg:v1
AMASS_PATH = "../datasets/skeleton_free/AMASS_P/processed"
SMPLH_PATH = "./smplh"
RIGNET_PATH = '../datasets/skeleton_free/RigNet'
CUSTOM_PATH = './skfree_workdir/train_3_6_rootsw_30part'
MIXAMO_PATH = '../datasets/skeleton_free/Mixamo'
MIXAMO_SIMPLIFY_PATH = '../datasets/skeleton_free/Mixamo_simplify'
# BLEND_FILE = './scene.blend'
# BLENDER_PATH = "\"C:/Program Files/Blender Foundation/Blender 2.83/blender.exe\""
TEMP_DIR = './skfree_workdir/train_3_6_rootsw_30part'
LOG_DIR = './skfree_workdir/train_3_6_rootsw_30part'

RADOM_SEED = 3047

os.makedirs(TEMP_DIR, exist_ok=True)

gdr2num = {'male': -1, 'neutral': 0, 'female': 1}
# MIXAMO_JOINTS = [ "Hips",   # 0
#   "LeftFoot",               # 1
#   "Spine1",                 # 2
#   "Spine2",                 # 3
#   "RightLeg",               # 4
#   "RightUpLeg",             # 5
#   "RightForeArm",           # 6
#   "LeftUpLeg",              # 7
#   "RightToeBase",           # 8
#   "LeftShoulder",           # 9
#   "RightHand",              # 10
#   "LeftForeArm",            # 11
#   "LeftArm",                # 12
#   "RightArm",               # 13
#   "RightFoot",              # 14
#   "Head",                   # 15
#   "LeftHand",               # 16
#   "RightShoulder",          # 17
#   "LeftLeg",                # 18
#   "Spine",                  # 19
#   "LeftToeBase",            # 20
#   "Neck"]                   # 21


MIXAMO_JOINTS = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
                "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
                "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
                "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                "RightShoulder", "RightArm", "RightForeArm", "RightHand"]
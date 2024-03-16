from os import listdir, makedirs
from os.path import exists, join

if __name__ == '__main__':
    obj_files_path = "/apdcephfs/private_jiaxuzhang_cq/datasets/skeleton_free/RigNet/npz"

    train_obj_split = "/apdcephfs/private_jiaxuzhang_cq/datasets/skeleton_free/RigNet/train_my_split.txt"
    test_obj_split = "/apdcephfs/private_jiaxuzhang_cq/datasets/skeleton_free/RigNet/test_my_split.txt"


    obj_file_lst = listdir(obj_files_path)
    obj_num = len(obj_file_lst)

    train_obj = obj_file_lst[:int(obj_num*0.7)]
    test_obj = obj_file_lst[int(obj_num*0.7):]

    with open(train_obj_split, 'w') as file:
        for obj in train_obj:
            file.write(obj.split(".")[0] + "\n")
    
    with open(test_obj_split, 'w') as file:
        for obj in test_obj:
            file.write(obj.split(".")[0] + "\n")
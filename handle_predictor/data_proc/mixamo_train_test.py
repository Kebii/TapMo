from os import listdir, makedirs
from os.path import exists, join

if __name__ == '__main__':
    obj_files_path = "./datasets/skeleton_free/Mixamo/obj"
    motion_file_path = "./datasets/skeleton_free/Mixamo_simplify/motion"

    train_obj_split = "./datasets/skeleton_free/Mixamo_simplify/train_char_split.txt"
    test_obj_split = "./datasets/skeleton_free/Mixamo_simplify/test_char_split.txt"

    train_motion_split = "./datasets/skeleton_free/Mixamo_simplify/train_motion_split.txt"
    test_motion_split = "./datasets/skeleton_free/Mixamo_simplify/test_motion_split.txt"

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
    

    motion_file_lst = listdir(motion_file_path)
    motion_num = len(motion_file_lst)

    train_motion = motion_file_lst[:int(motion_num*0.7)]
    test_motion = motion_file_lst[int(motion_num*0.7):]

    with open(train_motion_split, 'w') as file:
        for motion in train_motion:
            file.write(motion + "\n")
    
    with open(test_motion_split, 'w') as file:
        for motion in test_motion:
            file.write(motion + "\n")
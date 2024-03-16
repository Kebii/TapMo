import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from data_utils.mixamo_loader import MixamoDataset
from data_utils.rignet_loader import RignetDataset
from data_utils.amass_loader import AmassDataset


class MultiDataset(Dataset):
    # only used for training keypoint learning
    def __init__(self, amass_dir, mixamo_dir, rignet_dir,
                 smpl, part_augmentation=False, prob=(1/3, 1/3, 1/3), preload=None, single_part=True,
                 part_aug_scale=((0.5, 4), (0.6, 1), (0.3, 1.5)), simplify=True, new_rignet=True):
        # prob: probability of Amass data
        super(MultiDataset, self).__init__()
        if preload:
            self.amass, self.mixamo, self.rignet = preload
        else:
            if isinstance(part_augmentation, bool):
                p1 = p2 = p3 = part_augmentation
            else:
                p1, p2, p3 = part_augmentation
            self.amass = AmassDataset(amass_dir, smpl, part_augmentation=p1, simplify=simplify)
            self.mixamo = MixamoDataset(mixamo_dir, flag='train', part_augmentation=p2,
                                         single_part=single_part, part_aug_scale=part_aug_scale, joint_augmentation=False)
            if new_rignet:
                self.rignet = RignetDataset(rignet_dir, flag='train_my_split')
            else:
                self.rignet = RignetDataset(rignet_dir, flag='train_my_split')
        self.prob = prob

    def len(self):
        return 1000

    def database(self):
        return self.amass, self.mixamo, self.rignet

    def get(self, index):
        p = np.random.rand()
        if p <= self.prob[0]:
            return self.amass.get_uniform(np.random.randint(len(self.amass)))
        elif p <= self.prob[0] + self.prob[1] :
            return self.mixamo.get_uniform(np.random.randint(len(self.mixamo)))
        else:
            return self.rignet.get_uniform(np.random.randint(len(self.rignet)))

if __name__ == '__main__':
    from models.smpl import SMPL2Mesh
    from torch_geometric.loader import DataLoader
    from models.ops import *
    from global_var import *
    smpl = SMPL2Mesh(SMPLH_PATH)
    num_workers = 4
    part_aug_scale = ((0.5, 3), (0.7, 1), (0.5, 1.5))
    bs = 4
    train_set = MultiDataset(AMASS_PATH, MIXAMO_SIMPLIFY_PATH, RIGNET_PATH, smpl, part_aug_scale=part_aug_scale,
                             part_augmentation=(False, True, False), prob=(0.5, 0.4, 0.1),
                             single_part=True, simplify=True, new_rignet=True)
    train_loader = DataLoader(train_set, batch_size=bs,
                              shuffle=True, pin_memory=False, drop_last=True,
                              num_workers=num_workers)

    train_set2 = MultiDataset(AMASS_PATH, MIXAMO_SIMPLIFY_PATH, RIGNET_PATH, smpl, single_part=True,
                              part_aug_scale=part_aug_scale,
                              part_augmentation=(False, True, False), prob=(0.3, 0.4, 0.3),
                              preload=train_set.database(), simplify=True, new_rignet=True)
    train_loader2 = DataLoader(train_set2, batch_size=bs,
                               shuffle=True, pin_memory=False, drop_last=True,
                               num_workers=num_workers)
    
    for (data1, data2) in train_loader:
        print(data1.v0.shape)
        print(data2.v0.shape)
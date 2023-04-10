import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import transforms


class BoneData(Dataset):

    def __init__(self, root_dir, pose_dir):
        self.root_dir = root_dir
        self.label_dir = pose_dir
        self.path = os.path.join(root_dir, pose_dir)
        positive_list = os.listdir(os.path.join(self.path, "positive"))
        negative_list = os.listdir(os.path.join(self.path, "negative"))
        self.file_list = positive_list + negative_list
        self.positive_len = len(positive_list)
        self.negative_len = len(negative_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        label_dir = "positive" if idx < self.positive_len else "negative"
        file_path = os.path.join(self.path, label_dir, file_name)
        data = pd.read_csv(file_path, header=None, dtype="float32").to_numpy()
        label = np.float32(1) if label_dir == "positive" else np.float32(0)
        return data, label

    def __len__(self):
        return self.positive_len + self.negative_len

# root_dir = "DynamicData/Smash"
# label_dir = "positive"
# smash_ds = BoneData(root_dir, label_dir, transforms.ToTensor())
# print(len(smash_ds))
# print("1")

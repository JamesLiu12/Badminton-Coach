from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import transforms


class BoneData(Dataset):

    def __init__(self, root_dir, label_dir, transform):
        self.root_dir = root_dir
        self.label = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.filename_list = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx):
        file_name = self.filename_list[idx]
        file_path = os.path.join(self.path, file_name)
        data = pd.read_csv(file_path, header=None).to_numpy()
        data = self.transform(data)
        return data, 1 if label_dir == "positive" else 0

    def __len__(self):
        return len(self.filename_list)


root_dir = "Data/Smash"
label_dir = "positive"
smash_ds = BoneData(root_dir, label_dir, transforms.ToTensor())
print(len(smash_ds))
print("1")

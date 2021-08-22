import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import config


class ImageDS(Dataset):
    def __init__(self, root):
        super().__init__()
        self.data = []
        self.root = root
        self.class_names = os.listdir(root)

        for idx, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root, name))
            self.data += list(zip(files, [idx]*len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, label = self.data[idx]
        root_and_dir = os.path.join(self.root, self.class_names[label])
        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res

def test():
    dataset = ImageDS(root="Data/")
    loader = DataLoader(dataset, batch_size=1)
    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)

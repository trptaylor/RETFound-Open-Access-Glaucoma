import os
from PIL import Image
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

        self.samples = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                self.samples.append(os.path.join(folder, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, path

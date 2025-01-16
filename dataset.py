from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img




class MultiResolutionFolderDataset(Dataset):
    def __init__(self, root_path, transform, resolution=256):
        self.root_path = root_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.root_path, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img
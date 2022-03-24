import random
import torch
from torch.utils.data import Dataset
import glob
import os
import cv2
import numpy as np


class dataset(Dataset):
    def __init__(self, data_path):
        super(dataset, self).__init__()
        self.file = glob.glob(os.path.join(data_path, '*.tif'))
        assert len(self.file) > 0, '未找到数据!'

    def __getitem__(self, item):  # 获取数据
        image = self.file[item]
        label = self.file[item].replace('image', 'label')
        image = cv2.imread(image, 1)
        label = cv2.imread(label, -1)
        image, label = self.augment(image, label)
        return torch.tensor(image).type(torch.FloatTensor), torch.tensor(label).type(torch.FloatTensor)

    def augment(self, image, label):  # 数据增强:翻转
        image = image / 255
        label = label / 255
        image = np.transpose(image, (2, 0, 1))
        label = np.expand_dims(label, axis=0)
        if random.randint(0, 3) == 1:
            flip_mode = random.randint(-1, 1)
            image = cv2.flip(image, flip_mode)
            label = cv2.flip(label, flip_mode)
        return image, label

    def __len__(self):
        return len(self.file)

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import os
import numpy as np


class LaneClsDataset(torch.utils.data.Dataset):
    # 标注图片中 红色为背景类
    BACKGROUND_COLOR = np.array([0, 0, 255])
    LANE_COLOR = np.array([255, 0, 255])

    def __init__(self,
                 list_path,
                 dir_path,
                 img_shape=(800, 288),
                 num_classes=2):
        super(LaneClsDataset, self).__init__()
        self.dir_path = dir_path
        self.img_shape = img_shape
        self.list = [line.strip() for line in open(list_path)]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path, label_path = self.list[index].split()
        img_path = os.path.join(self.dir_path, img_path)
        label_path = os.path.join(self.dir_path, label_path)

        # get image
        image = cv2.resize(cv2.imread(img_path), self.img_shape)
        image = transforms.ToTensor()(image)

        # get mask data
        mask_image = cv2.resize(cv2.imread(label_path), self.img_shape)
        mask_bg = np.all(mask_image == self.LANE_COLOR, axis=2)
        mask_bg = mask_bg.reshape(*mask_bg.shape, 1)
        # print(mask_bg.shape)
        # mask_image = np.transpose(mask_bg, (2, 0, 1))
        mask_image = np.concatenate((mask_bg, np.invert(mask_bg)),
                                    axis=2).transpose((2, 0, 1))
        mask_image = torch.FloatTensor(mask_image)
        return image, mask_image


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self,
                 list_path,
                 dir_path,
                 img_shape=(800, 288)):
        super(LaneTestDataset, self).__init__()
        self.dir_path = dir_path
        self.img_shape = img_shape
        self.list = [line.strip() for line in open(list_path)]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.list[index])
        # get image
        image = cv2.resize(cv2.imread(img_path), self.img_shape)
        image = transforms.ToTensor()(image)

        return image, str(img_path)


if __name__ == "__main__":
    d = LaneClsDataset(list_path='../full.tsv',
                       dir_path='../data_road')


    image, label = d[1]
        
    label = label.numpy() * 255
    label = np.transpose(label, (1, 2, 0))
    print(label)
    img = Image.fromarray(np.uint8(label[:, :, 0]))
    img.show()
    cv2.waitKey(0)
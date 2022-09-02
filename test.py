import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from model.fcn import VGGNet, FCNs
from dataset.lane_cls_data import LaneTestDataset
import os
from tqdm import tqdm

CKPT_PATH = "./ckpt/epoch_60.pth"
OUT_PATH = "./output/"
IMG_H = 288
IMG_W = 800


def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fcn_model = FCNs(pretrained_net=VGGNet(pretrained=True))
    if device == 'cuda':
        fcn_model.to(device)
    fcn_model.load_state_dict(torch.load(CKPT_PATH))

    fcn_model.eval()

    test_set = LaneTestDataset(list_path='./test.tsv',
                               dir_path='./data_road',
                               img_shape=(IMG_W, IMG_H))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for image, image_path in tqdm(test_loader):
            image = image.to(device)
            output = fcn_model(image)
            output = torch.sigmoid(output)
            #Pytorch -> opencv structure
            mask = torch.argmax(output, dim=1).cpu().numpy().transpose((1, 2, 0))
            mask = mask.reshape(IMG_H, IMG_W)
            image = image.cpu().numpy().reshape(3, IMG_H, IMG_W).transpose((1, 2, 0)) * 255
            #####
            image[..., 2] = np.where(mask == 0, 255, image[..., 2])
            cv2.imwrite(os.path.join(OUT_PATH, os.path.basename(image_path[0])), image)


if __name__ == "__main__":
    main()

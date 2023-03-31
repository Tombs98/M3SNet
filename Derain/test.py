import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data2

from M3SNet import M3SNet
from M3SNet import M3SNetLocal
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results_M3SNetn_local_64/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./M3SNetn_64/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='HIDE', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = M3SNetLocal()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset, 'test', 'input')
test_dataset = get_test_data2(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir  = os.path.join(args.result_dir, dataset)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = data_test[0].cuda()
        filenames = data_test[1]
   

        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)


        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
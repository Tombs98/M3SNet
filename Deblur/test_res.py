
import os,sys,math
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from M3SNet import M3SNetLocal
import utils
from skimage.metrics import structural_similarity
from data_RGB import get_test_data, get_validation_data
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from M3SNet import M3SNet


def test():
    parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

    parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./res_msh/', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./M3SNetn_32_new/model_best.pth', type=str,
                      help='Path to weights')
    #parser.add_argument('--weights', default='M3SNet-GoPro-width32.pth', type=str,
     #                   help='Path to weights')                  
    parser.add_argument('--dataset', default='GoPr', type=str,
                        help='Test Dataset')  # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
    parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model_restoration = M3SNetLocal()
    utils.load_checkpoint(model_restoration,args.weights)
    #std = torch.load(args.weights)
    #torch.save({
     #           'state_dict': model_restoration.state_dict(),
      #          }, str("M3SNetn_32/deblur_model_best_32.pth"))
    #model_restoration.load_state_dict(std['params'])
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    dataset = args.dataset
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'test')
    # rgb_dir_test = os.path.join(args.input_dir, 'input')
    print(rgb_dir_test)
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)
    psnr_test = []
    ssim_test = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        target = data_test[1].cuda()
        with torch.no_grad():
            restored = model_restoration(input_)
        psnr_test.append(utils.torchPSNR(restored, target))
    psnr_val_rgb = torch.stack(psnr_test).mean().item()
    print(psnr_val_rgb)
        # print(ssim_test_rgb)
   



if __name__=='__main__':
    test()




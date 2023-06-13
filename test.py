import os
import cv2
import argparse
import torch
from MODELS.model import LLIE_Network as Model
from MODELS.model import SSIM
from MODELS.model import PSNR
from torchvision.utils import save_image as imwrite
from torchvision import transforms
import numpy as np


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--directory', type=str, default='./test_data')
parser.add_argument('--save', type=str, default='./MODELS/parameters_LOL.pth' , help= './MODELS/parameters_Syn.pth' or './MODELS/parameters_LOLV2_Real.pth'or'./MODELS/parameters_Rellisur.pth')
parser.add_argument('--Performance', type=str, default=False)
args = parser.parse_args()

transform=transforms.Compose([transforms.ToTensor()])
model=Model()
model=model.eval()
checkpoint = torch.load(args.save, map_location='cpu')
model.load_state_dict(checkpoint["state_dict"])
image_list=os.listdir(os.path.join(args.directory,'input'))

PSNR_list = []
ssim_list = []


for image in image_list:
    input = (cv2.imread(os.path.join(args.directory,'input',image))/255)[:,:,[2,1,0]].astype(np.float32())
    input = torch.unsqueeze(transform(input), dim=0)
    if args.Performance==True:
        output = (cv2.imread(os.path.join(args.directory,'output',image))/255)[:,:,[2,1,0]].astype(np.float32())
        output = torch.unsqueeze(transform(output), dim=0)

    with torch.no_grad():
        est=model(input)

    if args.Performance == True:
        PSNR_list.extend(PSNR(est, output))
        ssim_list.extend(SSIM(est, output))
    imwrite(est,'./test_result/'+image)
if args.Performance==True:
    avr_psnr = sum(PSNR_list) / len(PSNR_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    print('PSNR: '+str(avr_psnr))
    print('SSIM: ' + str(avr_ssim))






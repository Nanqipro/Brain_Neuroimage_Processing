import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import sys
import math
import scipy.io as scio
from network import Network_3D_Unet
from tensorboardX import SummaryWriter
import numpy as np
from data_process import shuffle_datasets, train_preprocess_lessMemoryMulStacks, shuffle_datasets_lessMemory
from utils import save_yaml
from skimage import io
#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--GPU', type=str, default="0", help="GPU index(es), e.g. '0' or '0,1,2,3'")

parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--img_s', type=int, default=150, help="the slices of image sequence")
parser.add_argument('--img_w', type=int, default=150, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=150, help="the height of image sequence")
parser.add_argument('--gap_s', type=int, default=60, help='the slices of image gap')
parser.add_argument('--gap_w', type=int, default=90, help='the width of image gap')
parser.add_argument('--gap_h', type=int, default=90, help='the height of image gap')

parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--normalize_factor', type=int, default=65535, help='normalize factor')

parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--datasets_folder', type=str, default='DataForPytorch', help="A folder containing files for training")
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--select_img_num', type=int, default=6000, help='select the number of images')
parser.add_argument('--train_datasets_size', type=int, default=1000, help='datasets size for training')
opt = parser.parse_args()

print('the parameter of your training ----->')
print(opt)
########################################################################################################################
if not os.path.exists(opt.output_dir): 
    os.mkdir(opt.output_dir)
current_time = opt.datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")
output_path = opt.output_dir + '/' + current_time
pth_path = 'pth//'+ current_time
if not os.path.exists(output_path): 
    os.mkdir(output_path)
if not os.path.exists(pth_path): 
    os.mkdir(pth_path)

yaml_name = pth_path+'//para.yaml'
save_yaml(opt, yaml_name)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
batch_size = opt.batch_size
lr = opt.lr

name_list, noise_img, coordinate_list = train_preprocess_lessMemoryMulStacks(opt)
# print('name_list -----> ',name_list)
########################################################################################################################
L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()
########################################################################################################################
denoise_generator = Network_3D_Unet(in_channels = 1,
                                out_channels = 1,
                                final_sigmoid = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    visible_gpu_count = torch.cuda.device_count()
    print(f'Using GPU. visible_gpu_count={visible_gpu_count}')
    denoise_generator = denoise_generator.to(device)
    L2_pixelwise = L2_pixelwise.to(device)
    L1_pixelwise = L1_pixelwise.to(device)
    if visible_gpu_count > 1:
        denoise_generator = nn.DataParallel(denoise_generator)
else:
    print('Using CPU.')
########################################################################################################################
optimizer_G = torch.optim.Adam( denoise_generator.parameters(), 
                                lr=opt.lr, betas=(opt.b1, opt.b2))
########################################################################################################################

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
prev_time = time.time()
########################################################################################################################
time_start=time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    name_list = shuffle_datasets_lessMemory(name_list)
    # print('name list -----> ',name_list)
    ####################################################################################################################     
    for index in range(0, len(name_list), batch_size):
        batch_names = name_list[index:index + batch_size]
        real_A_list = []
        real_B_list = []
        for n in batch_names:
            single_coordinate = coordinate_list[n]
            init_h = single_coordinate['init_h']
            end_h = single_coordinate['end_h']
            init_w = single_coordinate['init_w']
            end_w = single_coordinate['end_w']
            init_s = single_coordinate['init_s']
            end_s = single_coordinate['end_s']
            noise_patch1 = noise_img[init_s:end_s:2,init_h:end_h,init_w:end_w]
            noise_patch2 = noise_img[init_s+1:end_s:2,init_h:end_h,init_w:end_w]
            real_A_list.append(torch.from_numpy(noise_patch1).unsqueeze(0).unsqueeze(0))
            real_B_list.append(torch.from_numpy(noise_patch2).unsqueeze(0).unsqueeze(0))

        real_A = torch.cat(real_A_list, dim=0).to(device)
        real_B = torch.cat(real_B_list, dim=0).to(device)
        input_name = batch_names[-1]
        real_A = Variable(real_A)
        fake_B = denoise_generator(real_A)
        # Pixel-wise loss
        L1_loss = L1_pixelwise(fake_B, real_B)
        L2_loss = L2_pixelwise(fake_B, real_B)
        ################################################################################################################
        optimizer_G.zero_grad()
        # Total loss
        Total_loss =  0.5*L1_loss + 0.5*L2_loss
        Total_loss.backward()
        optimizer_G.step()
        ################################################################################################################
        samples_done = epoch * len(name_list) + min(index + len(batch_names), len(name_list))
        samples_left = opt.n_epochs * len(name_list) - samples_done
        time_left = datetime.timedelta(seconds=samples_left * (time.time() - prev_time))
        prev_time = time.time()
        ################################################################################################################
        if (index + len(batch_names)) % 50 == 0:
            time_end=time.time()
            print('time cost',time_end-time_start,'s \n')
            sys.stdout.write(
            "\r[Epoch %d/%d] [Sample %d/%d] [Total loss: %f, L1 Loss: %f, L2 Loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                min(index + len(batch_names), len(name_list)),
                len(name_list),
                Total_loss.item(),
                L1_loss.item(),
                L2_loss.item(),
                time_left,
            )
        )
        ################################################################################################################
    # if (epoch+1)%1 == 0:
        # torch.save(denoise_generator.state_dict(), pth_path + '//G_' + str(epoch) + '.pth')
        if (index + len(batch_names)) % 300 == 0:
            model_to_save = denoise_generator.module if isinstance(denoise_generator, nn.DataParallel) else denoise_generator
            torch.save(model_to_save.state_dict(), pth_path + '//G_' + str(epoch) +'_'+ str(index) + '.pth')
    if (epoch+1)%1 == 0:
        output_img = fake_B[0:1].cpu().detach().numpy()
        train_GT = real_B[0:1].cpu().detach().numpy()
        train_input = real_A[0:1].cpu().detach().numpy()
        image_name = input_name

        train_input = train_input.squeeze().astype(np.float32)*opt.normalize_factor
        train_GT = train_GT.squeeze().astype(np.float32)*opt.normalize_factor
        output_img = output_img.squeeze().astype(np.float32)*opt.normalize_factor
        train_input = np.clip(train_input, 0, 65535).astype('uint16')
        train_GT = np.clip(train_GT, 0, 65535).astype('uint16')
        output_img = np.clip(output_img, 0, 65535).astype('uint16')
        result_name = output_path + '/' + str(epoch) + '_' + str(index) + '_' + input_name+'_output.tif'
        noise_img1_name = output_path + '/' + str(epoch) + '_' + str(index) + '_' + input_name+'_noise1.tif'
        noise_img2_name = output_path + '/' + str(epoch) + '_' + str(index) + '_' + input_name+'_noise2.tif'
        io.imsave(result_name, output_img)
        io.imsave(noise_img1_name, train_input)
        io.imsave(noise_img2_name, train_GT)


model_to_save = denoise_generator.module if isinstance(denoise_generator, nn.DataParallel) else denoise_generator
torch.save(model_to_save.state_dict(), pth_path +'//G_' + str(opt.n_epochs) + '.pth')

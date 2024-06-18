import torch
from torchvision import transforms
from datetime import datetime
import os
import numpy as np
from datasets.crowd import Crowd
from models_v3.vgg import vgg19
import models
import models_v2
import models_v3
from glob import glob
import cv2
import random


# test_dir = '/home/teddy/crowd_data/Sh_A_Train_Val_NP/val'
# test_dir = r'E:\Dataset\UCF-Train-Val-Test\test'
test_dir = r'E:\Dataset\Counting\JHU_Train_Val_Test\test'
model_dir = r'model/JHU/vgg19-1217-155734'
model_name ='vgg19'
test_model_num = 6


# vis_dir = os.path.join(model_dir, 'vis_test')
# if not os.path.exists(vis_dir):
#     os.makedirs(vis_dir)
if __name__ == '__main__':

    model_list = sorted(glob(os.path.join(model_dir, '*.pth')))
    if len(model_list) > test_model_num:
        model_list = model_list[-test_model_num:]

    datasets = Crowd(test_dir, 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=4, pin_memory=True)
    torch.backends.cudnn.benchmark = False

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set vis gpu
    device = torch.device('cuda')

    # model = vgg19()
    model = getattr(models, model_name)()
    model.to(device)
    model.eval()

    # model.load_state_dict(torch.load(model_path, device))
    log_list = []
    for model_path in model_list:
        epoch_minus = []
        model.load_state_dict(torch.load(model_path, device))
        for inputs, count, name in dataloader:
            inputs = inputs.to(device)
            # print(inputs.size())
            # w,h = inputs.size(2), inputs.size(3)
            # if random.random()<0.5:
            # points1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            # points2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            # else:
            # points1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            # points2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            # M = cv2.getPerspectiveTransform(points1, points2)
            # inputs = cv2.warpPerspective(np.array(inputs[0]).transpose(1,2,0),M,(h,w))
            # if random.random()<0.5:
            # aaa=False
            # inputs = cv2.pyrDown(np.array(inputs[0]).transpose(1,2,0))
            # else:
            # aaa=True
            # inputs = cv2.pyrUp(np.array(inputs[0]).transpose(1,2,0))
            # inputs = inputs.transpose(2,0,1)
            # inputs = torch.tensor(inputs, dtype=torch.float32, device=device).unsqueeze(0)
            # print(inputs.size())
            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            with torch.set_grad_enabled(False):
                # if aaa:
                # outputs = model(inputs)
                # w,h = inputs.size(2), inputs.size(3)
                # outputs1 = model(inputs[:,:,:w//2,:h//2])
                # outputs2 = model(inputs[:,:,:w//2,h//2:])
                # outputs3 = model(inputs[:,:,w//2:,:h//2])
                # outputs4 = model(inputs[:,:,w//2:,h//2:])
                # flat_result = torch.flatten(outputs) * torch.flatten(mask)
                # temp_minu = count[0].item() - torch.sum(outputs).item()
                # temp_minu = count[0].item() - (torch.sum(outputs1)+torch.sum(outputs2)+torch.sum(outputs3)+torch.sum(outputs4)).item()

                outputs = model(inputs)
                temp_minu = count[0].item() - torch.sum(outputs).item()
                # print(name, temp_minu, count[0].item(), torch.sum(outputs).item(), torch.sum(outputs).item())
                epoch_minus.append(temp_minu)

        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = 'model_name {}, mae {}, mse {}'.format(os.path.basename(model_path), mae, mse)
        log_list.append(log_str)
        print(log_str)

    date_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    with open(os.path.join(model_dir, 'test_results_{}.txt'.format(date_str)), 'w') as f:
        for log_str in log_list:
            f.write(log_str + '\n')

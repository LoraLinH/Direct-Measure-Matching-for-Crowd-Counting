from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter, save_results
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np

from math import acos, ceil
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import models
from datasets.crowd import Crowd
from losses.weight_vector_ent_sink import Weight_Vec_Ent_Sink
from losses.weight_vector_ent_sink2 import Weight_Vec_Ent_Sink2

"""trainable sigama with consistency loss and sigma penalize"""
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[3]
    images2 = torch.stack(transposed_batch[1], 0)
    return images, images2, points, targets


def cal_l2(x, y, cood):
    x_dis = -2 * torch.matmul(x, cood) + x * x + cood * cood
    y_dis = -2 * torch.matmul(y, cood) + y * y + cood * cood
    y_dis.unsqueeze_(2)
    x_dis.unsqueeze_(1)
    dis = y_dis + x_dis
    return torch.sqrt(torch.relu(dis))


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            # assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")
        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}

        assert args.batch_size == 1
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=1,
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        self.model = getattr(models, args.model_name)()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.sigma = args.sigma
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.save_all  = args.save_all
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.criteron = nn.L1Loss()
        # self.criterion =  nn.MSELoss(reduction='sum')
        self.cood = torch.arange(0, args.crop_size, step=args.downsample_ratio,
                                 dtype=torch.float32,
                                 device=self.device) + args.downsample_ratio / 2.0
        self.cood.unsqueeze_(0)
        self.cood = self.cood / args.crop_size
        self.cood2 = torch.arange(0, 256, step=args.downsample_ratio,
                                 dtype=torch.float32,
                                 device=self.device) + args.downsample_ratio / 2.0
        self.cood2.unsqueeze_(0)
        self.cood2 = self.cood2 / 256
        self.crop_size = args.crop_size
        self.dis_mtx_density = self.cal_dis_density(self.cood)
        self.dis_mtx_density2 = self.cal_dis_density(self.cood2)
        self.dis_mtx_density12 = self.cal_dis_density12(self.cood2, self.cood)


    def cal_dis_mtx(self, points, cood):
        x = points[:, 0].unsqueeze_(1)
        y = points[:, 1].unsqueeze_(1)
        x = torch.clamp(x, min=0.0, max=1.0)
        y = torch.clamp(y, min=0.0, max=1.0)
        dis = cal_l2(x, y, cood)
        dis = dis.view((dis.size(0), -1))
        return dis

    def cal_dis_density(self, cood):
        x, y = torch.meshgrid(cood[0], cood[0])
        x = torch.clamp(x.flatten(), min=0.0, max=1.0)
        y = torch.clamp(y.flatten(), min=0.0, max=1.0)
        x_cood = x.unsqueeze(0)
        y_cood = y.unsqueeze(0)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)


        x_dis = (x - x_cood) ** 2
        y_dis = (y - y_cood) ** 2
        dis = y_dis + x_dis
        dis = torch.sqrt(torch.relu(dis))

        return dis

    def cal_dis_density12(self, cood, cood2):
        x, y = torch.meshgrid(cood[0], cood[0])
        x = torch.clamp(x.flatten(), min=0.0, max=1.0)
        y = torch.clamp(y.flatten(), min=0.0, max=1.0)
        x2, y2 = torch.meshgrid(cood2[0], cood2[0])
        x2 = torch.clamp(x2.flatten(), min=0.0, max=1.0)
        y2 = torch.clamp(y2.flatten(), min=0.0, max=1.0)
        x_cood = x2.unsqueeze(0)
        y_cood = y2.unsqueeze(0)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)


        x_dis = (x - x_cood) ** 2
        y_dis = (y - y_cood) ** 2
        dis = y_dis + x_dis
        dis = torch.sqrt(torch.relu(dis))

        return dis

    def cal_dis_points(self, points):
        x = points[:, 0] / self.crop_size
        y = points[:, 1] / self.crop_size
        x = torch.clamp(x, min=0.0, max=1.0)
        y = torch.clamp(y, min=0.0, max=1.0)
        x_cood = x.unsqueeze(0)
        y_cood = y.unsqueeze(0)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        x_dis = (x-x_cood) ** 2
        y_dis = (y-y_cood) ** 2
        dis = y_dis + x_dis
        dis = torch.sqrt(torch.relu(dis))

        return dis

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            # torch.cuda.empty_cache()
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()


    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_gnorm = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode
        # Iterate over data.
        for step, (inputs, inputs2, points, targets) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            targets = targets[0].to(self.device)
            points = points[0].to(self.device)
            inputs2 = inputs2.to(self.device)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                outputs2 = self.model(inputs2)
                flat_outputs = torch.flatten(outputs)
                flat_outputs2 = torch.flatten(outputs2)

                pre_count = torch.sum(outputs)
                pre_count2 = torch.sum(outputs2)
                gd_count = torch.sum(targets)


                if len(points) > 0:
                    dis_mtx_cross = self.cal_dis_mtx(points/self.crop_size, self.cood)
                    w_vector = Weight_Vec_Ent_Sink(dis_mtx_cross, self.dis_mtx_density, targets, self.sigma, self.device)

                    loss = w_vector(flat_outputs, targets)
                    # loss += 0.5 * torch.abs((pre_count - gd_count))
                    loss += 0.5 * self.sigma * ((pre_count - gd_count) ** 2)


                    w_vector2 = Weight_Vec_Ent_Sink2(self.dis_mtx_density, self.dis_mtx_density2, self.dis_mtx_density12, self.sigma, self.device)
                    loss += w_vector2(flat_outputs, flat_outputs2)
                    loss += 0.5 * self.sigma * ((pre_count - pre_count2) ** 2)

                    res = 0
                else:
                    loss = self.criteron(pre_count, gd_count)
                    res = loss
                    # grad_max = 0.0

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                g_res = (pre_count - gd_count).item()

                # epoch_loss_count.update(float(loss_count), N)
                epoch_loss.update(float(loss), N)
                epoch_gnorm.update(float(res), N)
                epoch_mse.update(np.mean(g_res * g_res), N)
                epoch_mae.update(np.mean(abs(g_res)), N)


            # if step % 200 == 0:
            #     print("loc loss {:.2f}, g res {:.2f} res {:.2f}".format(loss, abs(g_res), float(res)))

        logging.info('Epoch {} Loss Loc: {:.8f}, G_norm {:.2f} MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_gnorm.get_avg(),
                             np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models


    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            if h >= 4096 or w >= 4096:
                h_stride = int(ceil(1.0 * h / 4096))
                w_stride = int(ceil(1.0 * w / 4096))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))
        logging.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))

        model_state_dic = self.model.state_dict()
        if (mse + mae) < (self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))





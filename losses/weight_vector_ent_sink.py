from torch import nn
import torch
import numpy as np

def calculate(dis_map, f_cross, g_cross, alpha, beta, sigma):
    alpha_map = alpha.unsqueeze(0).expand(dis_map.size())
    alpha_map = alpha_map * beta[:,None]
    f_cross_map = f_cross.unsqueeze(0).expand(alpha_map.size())
    f_cross_map = f_cross_map + g_cross[:,None]
    total_map = torch.exp((f_cross_map - dis_map) / sigma)
    return torch.sum(total_map * alpha_map)

def c_transform(pow_dis, alpha):
    exp_pow_dis = torch.exp(-pow_dis)
    alpha_pow_dis = exp_pow_dis * alpha[:,None]
    sum_pow_dis = torch.sum(alpha_pow_dis, dim=0)
    c_tran = -torch.log(sum_pow_dis+1e-7)
    # c_tran = -sigma * torch.logsumexp(smooth_power_dis, dim=0)
    return c_tran

def c_transform_sta(pow_dis, alpha):
    max_pow = torch.max(-pow_dis, dim=0).values
    smooth_power_dis = -pow_dis - max_pow
    exp_dis = torch.exp(smooth_power_dis)
    alpha_pow_dis = exp_dis * alpha[:, None]
    sum_dis = torch.sum(alpha_pow_dis, dim=0)
    log_dis = torch.log(sum_dis + 1e-7)
    c_tran = -(log_dis + max_pow)
    return c_tran

def c_transform_ub(pow_dis, alpha):
    exp_dis = torch.exp(-pow_dis)
    sum_dis = torch.sum(exp_dis, dim=0)
    log_dis = torch.log(sum_dis + 1e-7)
    log_alpha_dis = torch.log(alpha + 1e-7)
    c_tran = log_alpha_dis - log_dis
    return c_tran

class Weight_Vec_Ent_Sink(nn.Module):
    def __init__(self, dis_mtx_cross, dis_mtx_density, target_mass, sigma, device):
        super(Weight_Vec_Ent_Sink, self).__init__()
        self.dis_mtx_cross = dis_mtx_cross/sigma
        self.dis_mtx_density = dis_mtx_density/sigma
        self.sigma = sigma
        self.device = device


    def forward(self, pre_density, target_mass, show_res=False):
        g_cross = torch.zeros(len(self.dis_mtx_cross), dtype=torch.float32, device=self.device)
        f_cross = torch.zeros(len(self.dis_mtx_density), dtype=torch.float32, device=self.device)
        p_auto = torch.zeros(len(self.dis_mtx_density), dtype=torch.float32, device=self.device)
        for _ in range(20):
            p_auto_pre = p_auto
            autof_pow_dis = self.dis_mtx_density - p_auto[:,None]
            p_auto = 0.5 * (p_auto + c_transform(autof_pow_dis, pre_density))
            if self.sigma * torch.mean(torch.abs(p_auto-p_auto_pre))< 1e-4:
                break

        for _ in range(200):
            g_cross_pre = g_cross
            f_cross_pre = f_cross
            crossg_pow_dis = self.dis_mtx_cross - g_cross[:,None]
            f_cross = c_transform(crossg_pow_dis, target_mass)
            f_cross = f_cross / (1 + self.sigma)
            crossf_pow_dis = self.dis_mtx_cross.transpose(1, 0) - f_cross[:,None]
            g_cross = c_transform(crossf_pow_dis, pre_density)
            if self.sigma * torch.mean(torch.abs(g_cross - g_cross_pre))< 1e-4:
                break

        autof_pow_dis = self.dis_mtx_density - p_auto[:, None]
        p_auto = c_transform(autof_pow_dis, pre_density)
        p_auto = p_auto * self.sigma
        g_cross = g_cross * self.sigma
        f_cross = f_cross * self.sigma
        f_cross = -torch.exp(-f_cross) + 1

        w_c_dis = torch.sum((f_cross - p_auto) * pre_density) + torch.sum(
              target_mass * g_cross)

        return w_c_dis * 1000


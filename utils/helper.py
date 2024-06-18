import os
import cv2
import numpy as np

class Save_Handle(object):
    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        sum_old = self.sum
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count

    def get_sum(self):
        return self.sum


def save_results(input, output,
                 output_dir, fname='results.png'):
    eps = 1e-5
    input = 255 * (input.detach().cpu().numpy()[0, 0, :, :] * 0.229 + 0.485)
    output = output.detach().cpu().numpy()[0, 0, :, :]
    output = 255 * output / (np.max(output) + eps)
    output = np.clip(output, 0, 255)
    h, w = output.shape
    # assert density_map.shape == output.shape
    output = cv2.resize(output, (input.shape[1], input.shape[0]))

    result_img = np.hstack((input, output))
    cv2.imwrite(os.path.join(output_dir, fname), result_img.astype(np.uint8))

def save_results_old(input, output, w_maps, prob_list,
                 output_dir, fname='results.png'):
    eps = 1e-5
    input = 255 * (input.detach().cpu().numpy()[0, 0, :, :] * 0.229 + 0.485)
    output = output.detach().cpu().numpy()[0, 0, :, :]
    output = 255 * output / (np.max(output) + eps)
    output = np.clip(output, 0, 255)
    h, w = output.shape
    # assert density_map.shape == output.shape
    output = cv2.resize(output, (input.shape[1], input.shape[0]))

    w_maps = w_maps.detach().cpu().numpy()[0, 0, :, :]
    w_maps = 255 * (w_maps - np.min(w_maps)) / (np.max(w_maps) + eps)
    w_maps = np.clip(w_maps, 0, 255)
    # assert density_map.shape == output.shape
    w_maps = cv2.resize(w_maps, (input.shape[1], input.shape[0]))

    result_img = np.hstack((input, output, w_maps))

    if prob_list[0] is not None:
        prob = np.max(prob_list[0].detach().cpu().numpy(), axis=0)
        prob = np.clip(np.reshape(prob, (h, w)) * 255, 0, 255)
        prob = cv2.resize(prob, (input.shape[1], input.shape[0]))
        result_img = np.hstack((result_img, prob))

    cv2.imwrite(os.path.join(output_dir, fname), result_img.astype(np.uint8))


def save_results_density(input, output, gd_density,
                         output_dir, fname='results.png'):
    eps = 1e-5
    input = 255 * (input.detach().cpu().numpy()[0, 0, :, :] * 0.229 + 0.485)
    output = output.detach().cpu().numpy()[0, 0, :, :]
    output = 255 * output / (np.max(output) + eps)
    output = np.clip(output, 0, 255)
    output = cv2.resize(output, (input.shape[1], input.shape[0]))

    if gd_density is not None:
        gd_density = gd_density.detach().cpu().numpy()[0, :, :]
        gd_density = 255 * gd_density / (np.max(gd_density) + eps)
        gd_density = np.clip(gd_density, 0, 255)
        gd_density = cv2.resize(gd_density, (input.shape[1], input.shape[0]))
        result_img = np.hstack((input, output, gd_density))
    else:
        result_img = np.hstack((input, output))

    cv2.imwrite(os.path.join(output_dir, fname), result_img.astype(np.uint8))

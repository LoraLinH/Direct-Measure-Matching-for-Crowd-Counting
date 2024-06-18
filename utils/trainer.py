import os
import logging
from datetime import datetime
from utils.logger import setlogger


class Trainer(object):
    def __init__(self, args):
        sub_dir = args.model_name + '-' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        self.vis_dir = os.path.join(self.save_dir, 'vis')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger

        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def train(self):
        """training one epoch"""
        pass

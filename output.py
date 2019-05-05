import os
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter


class OutputWriter(object):
    '''
    Used to directly print log to files
    (instead of waiting untill the end of the run to see print statements in the slurm file)
    These files can be made:
        log.txt in the base dir
        checkpoints in the checkpoint directory
        tensorboard files in the board directory
    '''
    def __init__(self, path_dir):
        self.dir = path_dir
        self.dir_check = os.path.join(path_dir, 'checkpoints')
        self.dir_board = os.path.join(path_dir, "board")
        os.makedirs(self.dir_check, exist_ok=True)
        os.makedirs(self.dir_board, exist_ok=True)

        self.writer = SummaryWriter(self.dir_board)
        
    def add_scalar(self, tag, scalar_value, global_step=None):
        '''
        Let summary writer add a scalar (same arguments as SummaryWriter)
        '''
        self.writer.add_scalar(tag, scalar_value, global_step)

    def write(self, file_name, text):
        '''
        Write string to line in text file
        '''
        with open(os.path.join(self.dir, file_name), 'a') as f:
            f.write(text+"\n")


    def save_model(self, model, iter):
        '''
        Save model state dict to pickle file
        '''
        torch.save(model, os.path.join(self.dir_check, '{:09d}.pt'.format(iter)))
        

    def log(self, text):
        '''
        Print and write to log file
        '''
        print(text)
        self.write('log.txt', text)
import os
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import models
import copy


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
        model_dict = {
            'has_elmo': model.embedding.has_elmo(),
            'has_glove': model.embedding.has_glove()
        }

        # Obtain embedding parameters
        if model.embedding.has_elmo():
            model_dict['elmo_params'] = model.embedding.elmo.get_mix_parameters()
        if model.embedding.has_glove():
            model_dict['glove_file'] = model.embedding.glove.glove_file

        model_copy = copy.deepcopy(model)
        model_copy.embedding.clear()
        model_dict['model'] = model_copy,
        
        torch.save(model_dict, os.path.join(self.dir_check, '{:09d}.pt'.format(iter)))

    def load_model(self, file, device='cuda'):
        '''
        Load a model that was saved by save_model()
        '''
        model_dict = torch.load(file)
        model = model_dict['model']

        # Add embeddings
        if model_dict['has_elmo']:
            model.embedding.set_elmo(mix_parameters=model_dict['elmo_params'])
        if model_dict['has_glove']:
            model.embedding.set_glove(model_dict['glove_file'])

        model.embedding.set_device(device)
        model.to(device)

        return model

        

    def log(self, text):
        '''
        Print and write to log file
        '''
        print(text)
        self.write('log.txt', text)
from torch import nn
from torch.nn.functional import relu
import torch
import re
import numpy as np
import os

from allennlp.modules.elmo import Elmo, batch_to_ids
from data import PennDataset, SnliDataset
from torch.utils.data import DataLoader
import eval

GLOVE_FILE = 'data/glove/glove.840B.300d.txt'
GLOVE_TRAIN_FILE = os.path.join('data', 'glove', 'glove_selection_snli-wic-wsj.pt') # file with GloVe vectors from all training data



class WordEmbedding(nn.Module):
    '''
    General module for word embeddings
    Supports contextualized ELMo embedding, GloVe vectors and index embedding
    '''

    def __init__(self, device='cpu'):
        super(WordEmbedding, self).__init__()
        # Embedding modules
        self.elmo = None
        self.glove = None
        # Device to which pytorch embeddings should be moved
        self.device = device
        self.to(device)

    def set_device(self, device):
        '''
        Sets device of each embedding (ELMo and/or Glove)
        '''
        self.device = device
        if self.has_elmo():
            self.elmo.set_device(device)
        elif self.has_glove():
            self.glove.set_device(device)

    def clear(self):
        '''
        Sets ELMo and GloVe to None
        Used for saving models
        '''
        self.elmo = None
        self.glove = None

    def has_elmo(self):
        return self.elmo is not None

    def has_glove(self):
        return self.glove is not None

    def set_elmo(self, mix_parameters=None):
        '''
        Add a contextual ELMo word embedding.
        Args:
            see ElmoEmbedding class
        '''
        self.elmo = ElmoEmbedding(mix_parameters=mix_parameters, device=self.device)
        # self.add_module("elmo", self.elmo)


    def set_glove(self, glove_file=GLOVE_TRAIN_FILE):
        '''
        Add a GloVe embedding module. 
        The embedding and vocabulary is taken from glove_file.
        Args:
            glove_file: pickle with with dict {embedding, w2i, i2w} that can be used directly as glove embedding
        '''
        self.glove = GloveEmbedding(glove_file=glove_file, device=self.device)

    def forward(self, batch):
        '''
        Embed a batch of sentences

        Args:
            batch: list of list of words to embed, e.g. [['First', 'sentence', '.'], ['Another', '.']]

        Returns:
            embeddings: tensor of concatenated words embeddings with dimensions (batch_size, sequence_length, embed_dim)
            TODO: lengths: sequence lengths for packing
        '''
        if self.has_elmo() and self.has_glove():
            return torch.cat([self.elmo(batch), self.glove(batch)], dim=2)
        elif self.has_elmo():
            return self.elmo(batch)
        elif self.has_glove():
            return self.glove(batch)
        else:
            raise Exception("WordEmbedding contains no ELMo and no GloVe")


class GloveEmbedding(nn.Module):

    def __init__(self, glove_file, device='cuda'):
        ''' 
        The embedding and vocabulary is taken from glove_file.
        Args:
            glove_file: pickle with with dict {embedding, w2i, i2w} that can be used directly as glove embedding
        '''
        super(GloveEmbedding, self).__init__()
        self.glove_file = glove_file

        # Get data from file
        glove_dict = torch.load(glove_file)
        self.w2i = glove_dict['w2i']
        self.i2w = glove_dict['i2w']
        embedding_weights = glove_dict['embedding']

        self.embedding = nn.Embedding.from_pretrained(embedding_weights).to(device)
        self.to(device)
        self.device = device
        
    def set_device(self, device):
        '''
        Sets device of embedding
        '''
        self.device = device
        self.to(device)

    def forward(self, batch):
        '''
        Embed a batch of sentences

        Args:
            batch: list of list of words to embed, e.g. [['First', 'sentence', '.'], ['Another', '.']]

        Returns:
            embeddings: tensor of embedded words with dimensions (batch_size, sequence_length, embed_dim=300)
                padded with <UNK> embeddings
        '''

        # Convert to tensor of indices (pad with 0 index)
        seq_len = max(len(sent) for sent in batch)
        indices = torch.LongTensor([[self.w2i.get(word, 0) for word in sent] + [0] * (seq_len - len(sent)) for sent in batch])

        # Embed
        return self.embedding(indices.to(self.device))


    @classmethod
    def make_selected_glove(cls, data, output_file, glove_file=GLOVE_FILE):
        '''
        Extracts 300D GloVe vectors of all lowercase words in data. 
        Args:
            data: list of lists of words
            output_file: output dict is written here
            glove_file: file with all GloVe vectors
        Output file:
            dict:
                w2i: dictionary mapping words to embedding indices, default is 0
                i2w: list mapping indices to words
                embedding: torch.FloatTensor, first rows represents <PAD> and is zero, second <UNK> and is average of all GloVe embeddings
        '''

        # Collect words (lowercase, filter only letters and numbers)
        words = list(set([re.sub(r"[^A-Za-z0-9]", '', word).lower() for sent in data for word in sent]))
        w2i = {'<PAD>': 0, '<UNK>': 1}
        i2w = ['<PAD>', '<UNK>']

        # Go through GloVe file add collect embeddings
        embeddings = []
        embeddingN = 2 # current index of added word
        embeddingTotalSum = np.zeros(300) # used to compute average embedding for <UNK>
        embeddingTotalN = 0 # count total number of embeddings in file
        with open(glove_file, 'r') as f:
            for line in f:
                # Obtain word and embedding
                word = "".join(line.split()[:-300])
                embedding = [float(x) for x in line.split()[-300:]]
                # Add embedding for <UNK> average embedding
                embeddingTotalSum += embedding
                embeddingTotalN += 1
                if embeddingTotalN % 219601 == 0:
                    print("{:02.1}%".format(embeddingTotalN / 2196017*100))
                if word in words:
                    # Add GloVe vector to embedding
                    embeddings.append(
                        [float(x) for x in line.split()[-300:]]
                    )
                    w2i[word] = embeddingN
                    i2w.append(word)
                    embeddingN += 1

        # Add average embedding for <UNK>
        averageEmbedding = embeddingTotalSum / embeddingTotalN
        embeddings = [np.zeros(300), averageEmbedding] + embeddings

        # Write to file
        out = {
            'w2i': w2i,
            'i2w': i2w,
            'embedding': torch.Tensor(embeddings)
        }
        torch.save(out, output_file)



class ElmoEmbedding(nn.Module):
    '''
    ELMo 5.5B model: trained on a dataset of 5.5B tokens consisting of Wikipedia (1.9B) and all of 
    the monolingual news crawl data from WMT 2008-2012 (3.6B).
    Model weights are fixed, mixing weights are either trained or fixed.
    '''
    def __init__(self, mix_parameters=None, device='cuda'):
        '''
        Args:
            mix_parameters: weights responsible for averaging between the character embedding 
                and the two LSTM states in that order; if None these weights are trained
        '''
        super(ElmoEmbedding, self).__init__()
        options_file = "data/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        weight_file = "data/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

        # initialise ELMo embedding
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0,
            requires_grad=False, scalar_mix_parameters=mix_parameters)
        
        # store device for embedding
        self.device = device
        self.to(device)
        
    def forward(self, batch):
        '''
        Embed a batch of sentences

        Args:
            batch: list of list of words to embed, e.g. [['First', 'sentence', '.'], ['Another', '.']]

        Returns:
            embeddings: tensor of embedded words with dimensions (batch_size, sequence_length, embed_dim=1024)
        '''

        # Convert words to character ids
        character_ids = batch_to_ids(batch).to(self.device)
        # Embed words
        elmo_out = self.elmo(character_ids)
        embeddings = elmo_out['elmo_representations'][0]
        # return batch embeddings
        return embeddings

    def get_mix_parameters(self):
        '''
        Returns the ELMo mix parameters in order (1) character-based word embedding, 
            (2) hidden state of first LSTM, (3) hidden state of second LSTM
        Overall scaling by gamma is accounted for in the individual parameters.
        To reconstruct the ELMo embedding, simply use these parameters as mix_parameters arrgument
            at initialisation
        '''
        gamma = self.elmo.scalar_mix_0.gamma.item()
        mix_parameters = [p.item() * gamma for p in self.elmo.scalar_mix_0.scalar_parameters]
        return mix_parameters

    def set_device(self, device):
        '''
        Sets device of embedding
        '''
        self.device = device
        self.to(device)

    def warm_up(self):
        '''
        Runs some batches to "warm up" (https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism)
        Recommended to do before evaluation or inference, so that results are more reproducable and constant
        '''
        dataset = PennDataset()
        for i in range(10):
            batch = dataset[i*100:(i+1)*100]
            self(batch)

def make_selected_glove_training():
    '''
    Makes a glove file containing all words that could be needed in training
    Fit for: SNLI, WiC
    '''

    # Collect words
    print("Collecting Words")
    words = []
    print("\tPenn Treebank POS")
    for set_name in ['train', 'dev', 'test']:
        dataset = PennDataset(set_name, first_label=False)
        ws = []
        for sent in dataset:
            ws += sent[0]
        words += list(set(ws))
        print("\t\t...")
    print("\tSNLI")
    for fname in ["snli_1.0_train.jsonl", "snli_1.0_dev.jsonl", "snli_1.0_test.jsonl"]:
        dataset = SnliDataset(os.path.join('data', 'snli', fname))
        ws = []
        for p in dataset:
            ws += p[1] + p[2]
        words += list(set(ws))
        data = None
        print("\t\t...")
    dataset = None
    print("\tWiC")
    wic = eval.WicEvaluator(None, None)
    for setname in ['train', 'dev', 'test']:
        data = wic.load_data(os.path.join(eval.PATH_TO_WIC, setname, setname+'.data.txt'))
        ws = []
        for p in data:
            ws += [w for sent in data for w in sent]
        words += list(set(ws))
        data = None
        print("\t\t...")

    # Make GloVe selection
    words = [list(set(words))]
    print("Selecting words from GloVe")
    GloveEmbedding.make_selected_glove(
        words, os.path.join('data', 'glove', 'glove_selection_snli-wic-wsj.pt'))
    

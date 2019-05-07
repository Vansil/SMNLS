from torch import nn
from torch.nn.functional import relu
import torch

from allennlp.modules.elmo import Elmo, batch_to_ids




class WordEmbedding(nn.Module):
    '''
    General module for word embeddings
    Supports contextualized ELMo embedding, GloVe vectors and index embedding
    TODO: index embedding: zero padding, maybe zero when index overflow; GloVe; add parameters to Module so optim can find them
    '''

    def __init__(self, device):
        super(WordEmbedding, self).__init__()
        # List of concatenated embedding modules
        self.embedding_modules = []
        # Device to which pytorch embeddings should be moved
        self.device = device
        self.to(device)

    def add_elmo(self, mix_parameters=None):
        '''
        Add a contextual ELMo word embedding.
        Args:
            see ElmoEmbedding class
        '''
        module = ElmoEmbedding(mix_parameters=mix_parameters, device=self.device)
        self.embedding_modules.append(module)
        self.add_module("elmo", module)


    def add_index_embedding(self, embed_dim, max_index):
        '''
        Add a trainable word index embedding.
        Args:
            see IndexEmbedding class
        '''
        module = IndexEmbedding(embed_dim, max_index, device=self.device)
        self.embedding_modules.append(module)
        self.add_module("index_embedding", module)

    def forward(self, batch):
        '''
        Embed a batch of sentences

        Args:
            batch: list of list of words to embed, e.g. [['First', 'sentence', '.'], ['Another', '.']]

        Returns:
            embeddings: tensor of concatenated words embeddings with dimensions (batch_size, sequence_length, embed_dim)
            TODO: lengths: sequence lengths for packing
        '''
        return torch.cat(
            [embed(batch) for embed in self.embedding_modules],
            dim=2
        )


class IndexEmbedding(nn.Module):
    '''
    Index embedding. Embeds words in a sentence based solely on their position in it.
    '''
    def __init__(self, embed_dim, max_index, device='cuda'):
        '''
        Args:
            embed_dim: dimension of each index embedding
            max_index: maximum supported index number
        '''
        super(IndexEmbedding, self).__init__()
        self.device = device
        self.embed = nn.Embedding(max_index, embed_dim)
        self.max_index = max_index
        self.to(device)
        
    def forward(self, batch):
        '''
        Embed a batch of sentences, that is: give the embeddings of index 0 to seqlen for each sample in the batch.

        Args:
            batch: list of list. Second list could contain words, e.g. [['First', 'sentence', '.'], ['Another', '.']]
                    but the actual content doesn't matter

        Returns:
            embeddings: tensor of embedded indices with dimensions (batch_size, sequence_length, embed_dim=1024)
        '''
        # Construct array of indices
        sequence_length = len(batch[0])
        batch_size = len(batch)
        # repeat max index if number of indices is too large for embedding
        n_index_overflow = max(0, sequence_length - self.max_index)
        if n_index_overflow > 0:
            print("WARNING: Index embedding requested for index out of scope (index {}, max index {})".format(sequence_length-1, self.max_index-1))
            indices = batch_size * [[ind for ind in range(self.max_index)] + n_index_overflow * [self.max_index-1]]
        else:
            indices = batch_size * [[ind for ind in range(sequence_length)]]
        indices_torch = torch.LongTensor(indices).to(self.device)

        # Embed indices
        embeddings = self.embed(indices_torch)
        return embeddings


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
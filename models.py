from torch import nn
from torch.nn.functional import relu
import torch

from embeddings import WordEmbedding


class TestModel(nn.Module):
    '''
    model to test SentEval, returns random word and sentence embedding
    '''
    def __init__(self):
        super(TestModel, self).__init__()
        self.emb = nn.Embedding(1,10)

    def forward(self, x):
        batch_size = len(x)
        return torch.rand(batch_size)

    def embed_sentences(self, batch):
        '''
        Embeds each sentence in the batch by averaging ELMo embeddings.
        NOTE: not used in training, only for sentence embedding evaluation
        Args:
            batch: list of list of words from premise sentences, e.g. [['First', 'sentence', '.'], ['Another', '.']]
        Returns:
            embedded: sentence embeddings. Shape (batch, features)
        '''
        return self.embed_words(batch).mean(dim=1)

    def embed_words(self, batch):
        '''
        Embeds each word in a batch of sentences using ELMo embeddings (contextualized)
        Args:
            batch: list of list of words from premise sentences, e.g. [['First', 'sentence', '.'], ['Another', '.']]
        Returns:
            embedded: ELMo embedding of batch, padded to make sentences of equal length. Shape (batch, sequence, features)
        '''
        batch_size = len(batch)
        seq_len = len(batch[0])
        indices = torch.LongTensor(batch_size * [seq_len * [0]])
        return self.emb(indices)
    



class BaseModelElmo(nn.Module):
    '''
    Model to test elmo embedding
    '''
    def __init__(self, input_size, hidden_size, output_size, device='cuda'):
        super(BaseModelElmo, self).__init__()
        self.elmo = WordEmbedding(device=device)
        self.elmo.add_elmo()

        self._l1 = nn.Linear(input_size * 4, hidden_size)
        self._l2 = nn.Linear(hidden_size, output_size)

        self.to(device)

    def forward(self, X1, X2):
        '''
        Args:
            X1: list of list of words from premise sentences, e.g. [['First', 'sentence', '.'], ['Another', '.']]
            X2: same as X1 for hypothesis sentences
        '''

        # embed together such that padding is same for both sentences
        elmos = self.embed_words(X1 + X2)
        # separate
        batch_size = len(X1)
        elmos1 = elmos[:batch_size, :]
        elmos2 = elmos[batch_size:, :]

        # embedding: mean of word embeddings
        E1 = elmos1.mean(dim=1)
        E2 = elmos2.mean(dim=1)

        # Combine sentences for classification
        abs_diff = torch.abs(E1 - E2)
        elem = E1 * E2
        concat = torch.cat([E1, E2, abs_diff, elem], dim=1)

        # Classify
        return self._classify(concat)
        
    def _classify(self, X):
        X = self._l1(X)
        X = relu(X)
        return self._l2(X)

    def embed_sentences(self, batch):
        '''
        Embeds each sentence in the batch by averaging ELMo embeddings.
        NOTE: not used in training, only for sentence embedding evaluation
        Args:
            batch: list of list of words from premise sentences, e.g. [['First', 'sentence', '.'], ['Another', '.']]
        Returns:
            embedded: sentence embeddings. Shape (batch, features)
        '''
        word_embed = self.embed_words(batch)
        return word_embed.mean(dim=1)

    def embed_words(self, batch):
        '''
        Embeds each word in a batch of sentences using ELMo embeddings (contextualized)
        Args:
            batch: list of list of words from premise sentences, e.g. [['First', 'sentence', '.'], ['Another', '.']]
        Returns:
            embedded: ELMo embedding of batch, padded to make sentences of equal length. Shape (batch, sequence, features)
        '''
        return self.elmo(batch)




class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseModel, self).__init__()
        self._l1 = nn.Linear(input_size * 4, hidden_size)
        self._l2 = nn.Linear(hidden_size, output_size)

    def forward(self, X1, x1_lengths, X2, x2_lengths):
        E1 = self._embed(X1, x1_lengths)
        E2 = self._embed(X2, x2_lengths)

        abs_diff = torch.abs(E1 - E2)
        elem = E1 * E2

        concat = torch.cat([E1, E2, abs_diff, elem], dim=1)

        return self._classify(concat)

    def _classify(self, X):
        X = self._l1(X)
        X = relu(X)
        return self._l2(X)

    def _embed(self, X, lengths):
        embedding = torch.empty([X.size(0), X.size(2)], dtype=X.dtype).to(X.device)

        for i in range(X.size(0)):
            embedding[i, :] = torch.mean(X[i, 0:lengths[i], :], dim=0)

        return embedding

class LstmModel(BaseModel):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(LstmModel, self).__init__(embedding_size, hidden_size, output_size)

        self._lstm_module = nn.LSTM(input_size, embedding_size, 1, bidirectional=False, dropout=0, batch_first=True)

    def _embed(self, X, lengths):
        bsize = X.size(0)

        lengths_sorted, idx_sort = torch.sort(lengths)
        inv_idx = torch.arange(lengths.size(0)-1, -1, -1).long().to(lengths.device)
        lengths_sorted = lengths_sorted.index_select(0, inv_idx)
        idx_sort = idx_sort.index_select(0, inv_idx)
        _, idx_unsort = torch.sort(idx_sort)

        X = X.index_select(0, idx_sort)

        X_packed = nn.utils.rnn.pack_padded_sequence(X, lengths_sorted, batch_first=True)

        output = self._lstm_module(X_packed)[0]

        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        output = output.index_select(0, idx_unsort)

        output = output[(list(range(output.size(0))), lengths - 1)]

        return output

class BiLstmModel(BaseModel):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(BiLstmModel, self).__init__(2 * embedding_size, hidden_size, output_size)

        self._lstm_module = nn.LSTM(input_size, embedding_size, 1, bidirectional=True, dropout=0, batch_first=True)
        self._embedding_size = embedding_size

    def _embed(self, X, lengths):
        bsize = X.size(0)

        lengths_sorted, idx_sort = torch.sort(lengths)
        inv_idx = torch.arange(lengths.size(0)-1, -1, -1).long().to(lengths.device)
        lengths_sorted = lengths_sorted.index_select(0, inv_idx)
        idx_sort = idx_sort.index_select(0, inv_idx)
        _, idx_unsort = torch.sort(idx_sort)

        X = X.index_select(0, idx_sort)

        X_packed = nn.utils.rnn.pack_padded_sequence(X, lengths_sorted, batch_first=True)

        output = self._lstm_module(X_packed)[0]

        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        output = output.index_select(0, idx_unsort)

        output_fw = output[(list(range(output.size(0))), lengths - 1)]
        output_fw = output_fw[:, :self._embedding_size]
        output_bw = output[:, 0, self._embedding_size:]

        return torch.cat([output_fw, output_bw], dim=1)

class BiLstmMaxModel(BaseModel):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(BiLstmMaxModel, self).__init__(2 * embedding_size, hidden_size, output_size)

        self._lstm_module = nn.LSTM(input_size, embedding_size, 1, bidirectional=True, dropout=0, batch_first=True)

    def _embed(self, X, lengths):
        bsize = X.size(0)

        lengths_sorted, idx_sort = torch.sort(lengths)
        inv_idx = torch.arange(lengths.size(0)-1, -1, -1).long().to(lengths.device)
        lengths_sorted = lengths_sorted.index_select(0, inv_idx)
        idx_sort = idx_sort.index_select(0, inv_idx)
        _, idx_unsort = torch.sort(idx_sort)

        X = X.index_select(0, idx_sort)

        X_packed = nn.utils.rnn.pack_padded_sequence(X, lengths_sorted, batch_first=True)

        output = self._lstm_module(X_packed)[0]

        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        output = output.index_select(0, idx_unsort)

        output = torch.max(output, dim=1)[0]

        return output


def count_parameters(model):
    '''
    Computes the total number of parameters, and the number of trainable parameters
    Args:
        model: pytorch model
    Returns:
        num_params: total number of parameters
        num_trainable: number of trainable parameters
    '''
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, num_trainable
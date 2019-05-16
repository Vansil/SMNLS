from torch import nn
from torch.nn.functional import relu
import torch
import os

from embeddings import WordEmbedding, ElmoEmbedding, GloveEmbedding

GLOVE_TRAIN_FILE = os.path.join('data', 'glove', 'glove_selection_snli-wic-wsj.pt') # file with GloVe vectors from all training data


class BaselineElmo(nn.Module):
    '''
    Baseline model as in WiC paper by Pilehvar & Camacho-Collados
    Returns hidden state of the first ELMo LSTM
    '''
    def __init__(self, mix_parameters=[1/3, 1/3, 1/3]):
        super(BaselineElmo, self).__init__()
        # make embedding
        self.embedding = WordEmbedding()
        self.embedding.set_elmo(mix_parameters=mix_parameters)

    def forward(self, batch):
        return self.embed_words(batch)

    def embed_words(self, batch):
        return self.embedding(batch)

    def embed_sentences(self, batch):
        raise Exception("ELMo1 does not produce a sentence embedding")

class TestModelEmbedding(nn.Module):
    '''
    Model to test elmo and glove embedding
    Task: NLI
    '''
    def __init__(self, hidden_size, output_size, device='cuda'):
        super(TestModelEmbedding, self).__init__()
        self.embedding = WordEmbedding()
        self.embedding.set_elmo()
        self.embedding.set_glove(GLOVE_TRAIN_FILE)

        self._l1 = nn.Linear(1324 * 4, hidden_size)
        self._l2 = nn.Linear(hidden_size, output_size)

        self.to(device)

    def forward(self, X1, X2):
        '''
        Args:
            X1: list of list of words from premise sentences, e.g. [['First', 'sentence', '.'], ['Another', '.']]
            X2: same as X1 for hypothesis sentences
        '''

        # embedding: mean of word embeddings
        E1 = self.embed_sentences(X1)
        E2 = self.embed_sentences(X2)

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
        return self.embedding(batch)



class TestModel(nn.Module):
    '''
    model to test SentEval, returns random word and sentence embedding
    '''
    def __init__(self, device='cuda'):
        super(TestModel, self).__init__()
        self.emb = nn.Embedding(1,10)
        self.device = device
        self.to(device)

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
        indices = torch.LongTensor(batch_size * [seq_len * [0]]).to(self.device)
        return self.emb(indices)


class EmbeddingModule(nn.Module):
    def __init__(self, output_size):
        super(EmbeddingModule, self).__init__()
        self._output_size = output_size

    @property
    def output_size(self):
        return self._output_size

    def embed(self, input, lengths):
        raise NotImplementedError()


class VuaSequenceModel(EmbeddingModule):
    def __init__(self, embedding_module, num_classes=2, lstm_hidden_size=512):
        super(VuaSequenceModel, self).__init__(embedding_module.output_size + 2 * lstm_hidden_size)

        self._embedding_module = embedding_module
        self._lstm_layer = nn.LSTM(self._embedding_module.output_size, lstm_hidden_size, num_layers=1, bidirectional=True, dropout=0, batch_first=True)
        self._classifier = nn.Linear(2 * lstm_hidden_size, num_classes)

    def forward(self, input, lengths):
        lower_embedding = self._embedding_module.embed(input, lengths)

        packed = nn.utils.rnn.pack_padded_sequence(lower_embedding, lengths, batch_first=True, enforce_sorted=False)

        output = self._lstm_layer(packed)[0]

        unpacked = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        return self._classifier(unpacked)
        

    def embed(self, input, lengths):
        lower_embedding = self._embedding_module.embed(input, lengths)

        packed = nn.utils.rnn.pack_padded_sequence(lower_embedding, lengths, batch_first=True, enforce_sorted=False)

        output = self._lstm_layer(packed)[0]

        unpacked = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        return torch.cat([lower_embedding, unpacked], dim=2)

    def state_dict(self):
        return self._lstm_layer.state_dict()

    def load_state_dict(self, state_dict):
        self._lstm_layer.load_state_dict(state_dict)


class SnliModel(EmbeddingModule):
    def __init__(self, embedding_module, num_classes=3, lstm_hidden_size=512):
        super(SnliModel, self).__init__(embedding_module.output_size + 2 * lstm_hidden_size)

        self._embedding_module = embedding_module
        self._lstm_layer = nn.LSTM(self._embedding_module.output_size, lstm_hidden_size, num_layers=1, bidirectional=True, dropout=0, batch_first=True)
        self._classifier = nn.Sequential(
            nn.Linear(2 * 4 * lstm_hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def embed(self, input, lengths):
        lower_embedding = self._embedding_module.embed(input, lengths)

        packed = nn.utils.rnn.pack_padded_sequence(lower_embedding, lengths, batch_first=True, enforce_sorted=False)

        output = self._lstm_layer(packed)[0]

        unpacked = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        return torch.cat([lower_embedding, unpacked], dim=2)

    def forward(self, s1, s1_lengths, s2, s2_lengths):
        E1 = self._embedding_module.embed(s1, s1_lengths)
        E2 = self._embedding_module.embed(s2, s2_lengths)

        packed1 = nn.utils.rnn.pack_padded_sequence(E1, s1_lengths, batch_first=True, enforce_sorted=False)
        packed2 = nn.utils.rnn.pack_padded_sequence(E2, s2_lengths, batch_first=True, enforce_sorted=False)
        
        output1 = self._lstm_layer(packed1)[0]
        output2 = self._lstm_layer(packed2)[0]

        unpacked1 = nn.utils.rnn.pad_packed_sequence(output1, batch_first=True)[0]
        unpacked2 = nn.utils.rnn.pad_packed_sequence(output2, batch_first=True)[0]

        X1, _ = torch.max(unpacked1, dim=1)
        X2, _ = torch.max(unpacked2, dim=1)

        abs_diff = torch.abs(X1 - X2)
        elem = X1 * X2

        c = torch.cat([X1, X2, abs_diff, elem], dim=1)

        return self._classifier(c)


class WordEmbeddingModel(EmbeddingModule):
    def __init__(self, device, use_glove=True, use_elmo=True):
        glove_size = 300 if use_glove else 0
        elmo_size = 1024 if use_elmo else 0
        super(WordEmbeddingModel, self).__init__(glove_size + elmo_size)

        if not use_glove and not use_elmo:
            raise ValueError("Should use at least one form of embedding.")

        if use_elmo:
            self._elmo = ElmoEmbedding(device=device)
        if use_glove:
            self._glove = GloveEmbedding(GLOVE_TRAIN_FILE, device=device)

        self._use_elmo = use_elmo
        self._use_glove = use_glove

    def forward(self, sentences, lengths):
        if self._use_elmo and self._use_glove:
            return torch.cat([self._elmo(sentences), self._glove(sentences)], dim=2)
        elif self._use_elmo:
            return self._elmo(sentences)
        else:
            return self._glove(sentences)

    def embed(self, sentences, lengths):
        return self.forward(sentences, lengths)


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

def model_state_dict(model):
    '''
    Filters the state dict of a model to exclude the embedding.
    The embedding in the model should have the name 'embedding'
    '''
    return {key: value for key, value in model.state_dict().items() if key.split('.')[0] != 'embedding'}
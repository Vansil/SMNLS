from torch import nn
from torch.nn.functional import relu
import torch


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

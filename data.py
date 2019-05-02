from torch.utils.data import Dataset
from json import loads
import re
import numpy as np
import torch
import os
# from tqdm import tqdm
import pickle

class SnliDataset(Dataset):
    '''
    Dataset that returns SNLI items as a list of words as string
    '''
    def __init__(self, json_file):
        '''
        Args:
            json_file: path to SNLI json file
        '''
        data = []
        valid_chars_regex = r"[^a-zA-Z ]" # characters that will be filtered from sentences
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data_point = self.extract_data(loads(line), valid_chars_regex)
                if data_point is not None:
                    data.append(data_point)

        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        datum = self._data[idx]
        return datum["label"], datum["premise"], datum["hypothesis"]

    def extract_data(self, data_point, valid_chars_regex):
        '''
        Returns dictionary with label (0,1,2), premise and hypothesis (list of words)
        Returns None if annotators don't agree
        '''
        # Return None if there is no annotator agreement
        if data_point["gold_label"] == "-":
            return None
        
        # Extract info
        label = label_dict[data_point["gold_label"]]
        sent_premise = re.sub(valid_chars_regex, "", data_point["sentence1"]).split()
        sent_hypothesis = re.sub(valid_chars_regex, "", data_point["sentence2"]).split()

        return {
            "label": label,
            "premise": sent_premise,
            "hypothesis": sent_hypothesis
        }

class SnliDataLoader(object):
    '''Dataloader for Snli data'''

    def __init__(self, dataset, batch_size=64):
        # Store dataset
        self._dataset = dataset
        # Vars for sampling
        self._index_in_epoch = 0
        self._num_examples = len(dataset)
        self._epochs_completed = 0
        self._batch_size = batch_size
        assert batch_size <= self._num_examples, "batch size {} is larger than data size {}".format(batch_size, self._num_examples)
        # Determine random order for sampling
        self._sample_order = np.arange(self._num_examples)
        np.random.shuffle(self._sample_order)

    def __iter__(self):
        # uses itself as iterator
        return self

    def __next__(self):
        '''
        Sample the next batch
        
        Args:
            batch_size: number of data points in the batch

        Returns:
            labels: label of each point
            premises: list of premise sentences as word list
            hypotheses: list of hypothesis sentences as word list
        '''
        # Determine start and end data index
        start = self._index_in_epoch
        self._index_in_epoch += self._batch_size
        if self._index_in_epoch > self._num_examples:
            # End of dataset is reached, stop iterating and reshuffle dataset
            self._epochs_completed += 1
            self._index_in_epoch = self._batch_size

            self._sample_order = np.arange(self._num_examples)
            np.random.shuffle(self._sample_order)

            raise StopIteration
        end = self._index_in_epoch
        # select data
        indices = [self._sample_order[i] for i in range(start,end)]
        data = [self._dataset[ind] for ind in indices]
        labels = torch.LongTensor([d[0] for d in data])
        premises = [d[1] for d in data]
        hypotheses = [d[2] for d in data]
        
        return labels, premises, hypotheses


class SnliDataset_(Dataset):
    '''
    DEPRECATED
    '''
    def __init__(self, json_file, embeddings):
        with open(json_file, "r", encoding="utf-8") as f:
            data = [loads(line) for line in f.readlines()]

        self._embeddings = embeddings

        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        datum = self._data[idx]

        item = _extract_data(datum, self._embeddings)

        return item["label"], item["s1"], item["s2"]

label_dict = {
    "neutral": 0,
    "contradiction": 1,
    "entailment": 2
}

splitting_regex = re.compile(r"[^() ]")

def _extract_data(data_point, embeddings):
    s1 = splitting_regex.findall(data_point["sentence1_binary_parse"])
    s2 = splitting_regex.findall(data_point["sentence2_binary_parse"])
    if data_point["gold_label"] != "-":
        label = label_dict[data_point["gold_label"]]
    else:
        label = label_dict[data_point["annotator_labels"][0]]

    return {
        "label": label,
        "s1": torch.from_numpy(np.stack([embeddings[token] for token in s1])),
        "s2": torch.from_numpy(np.stack([embeddings[token] for token in s2]))
    }

def load_embeddings(file_path):
    pickle_path = file_path + ".pkl"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    else:
        embeddings = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                try:
                    tokens = line.split()
                    embeddings[tokens[0]] = np.array([float(token) for token in tokens[1:]], dtype=np.float32)
                except ValueError:
                    pass

        with open(pickle_path, "wb") as f:
            pickle.dump(embeddings, f)

        return embeddings

def collate_fn(batch):
    labels = []
    first_sentences = []
    second_sentences = []

    for label, s1, s2 in batch:
        labels.append(label)
        first_sentences.append(s1)
        second_sentences.append(s2)

    labels = torch.LongTensor(labels)

    l1 = [len(s) for s in first_sentences]
    l2 = [len(s) for s in second_sentences]

    batch_size = len(batch)
    m1 = max(l1)
    m2 = max(l2)

    fs = first_sentences[0].size(1)

    s1 = torch.zeros(batch_size, m1, fs, dtype=first_sentences[0].dtype)
    s2 = torch.zeros(batch_size, m2, fs, dtype=second_sentences[0].dtype)

    for i, s in enumerate(first_sentences):
        s1[i, 0:s.size(0), :] = s

    for i, s in enumerate(second_sentences):
        s2[i, 0:s.size(0), :] = s

    return labels, s1, torch.LongTensor(l1), s2, torch.LongTensor(l2)

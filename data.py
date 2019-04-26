from torch.utils.data import Dataset
from json import loads
import re
import numpy as np
import torch
import os
# from tqdm import tqdm
import pickle

class SnliDataset(Dataset):
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

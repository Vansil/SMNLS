from torch.utils.data import Dataset
from json import loads
import re
import numpy as np
import torch
import os
# from tqdm import tqdm
import pickle
from csv import DictReader
import xml.etree.ElementTree as ET

import eval

PENN_TREEBANK_PATH = "data/penn/wsj/"


class SnliDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = [loads(line) for line in f.readlines()]

        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        datum = self._data[idx]

        item = _extract_data(datum)

        return item["label"], item["s1"], item["s2"]

label_dict = {
    "neutral": 0,
    "contradiction": 1,
    "entailment": 2
}

splitting_regex = re.compile(r"[^() ]+")

def _extract_data(data_point):
    s1 = splitting_regex.findall(data_point["sentence1_binary_parse"])
    s2 = splitting_regex.findall(data_point["sentence2_binary_parse"])
    if data_point["gold_label"] != "-":
        label = label_dict[data_point["gold_label"]]
    else:
        label = label_dict[data_point["annotator_labels"][0]]

    return {
        "label": label,
        "s1": s1,
        "s2": s2
    }

def snli_collate_fn(batch):
    labels = []
    first_sentences = []
    second_sentences = []

    for label, s1, s2 in batch:
        labels.append(label)
        first_sentences.append(s1)
        second_sentences.append(s2)

    l1 = [len(s) for s in first_sentences]
    l2 = [len(s) for s in second_sentences]

    return first_sentences, torch.LongTensor(l1), second_sentences, torch.LongTensor(l2), torch.LongTensor(labels)


class VuaSequenceDataset(Dataset):
    def __init__(self, split="train"):
        super(VuaSequenceDataset, self).__init__()
        self._data = self._read_file(split)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]

        words = item["sentence"].split()
        labels = torch.LongTensor([int(l) for l in item["label_seq"][1:-1].split(", ")])

        num_metaphors = torch.sum(labels == 1).item()

        idxs = (labels == 0).nonzero()

        total_number =  (idxs.size(0) - num_metaphors + torch.randint(-1, 2, (2,))).item()

        total_number = max(0, min(total_number, idxs.size(0)))

        perm = torch.randperm(idxs.size(0))

        labels[idxs[perm[:total_number]]] = -100

        return words, labels

    def _read_file(self, split):
        with open(os.path.join("data", "vua-sequence", f"{split}.csv"), "r", encoding="ISO-8859-1") as f:
            reader = DictReader(f, fieldnames=["txt_id", "sen_ix", "sentence", "label_seq", "pos_seq", "labeled_sentence", "genre"])

            data = list(reader)[1:]

            return data


def vua_sequence_collate_fn(batch):
    sentences = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    max_len = max(len(s) for s in sentences)

    lengths = torch.LongTensor([len(s) for s in sentences])

    l = -torch.ones(len(labels), max_len, dtype=torch.int64) * 100

    for i, label in enumerate(labels):
        l[i, 0:len(label)] = label

    return sentences, lengths, l


class WordSenseDataset(Dataset):
    def __init__(self, split="train"):
        super(WordSenseDataset, self).__init__()

        self._data = [sentences for file in word_sense_files for sentences in self._load_file_data(file)]

        classes = sorted(list(set(word.attrib["sense"] for sentence in self._data for word in sentence if "sense" in word.attrib)))

        self._class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sentence = self._data[idx]

        words = [word.attrib["text"] for word in sentence]
        classes = [word.attrib["sense"] if "sense" in word.attrib else None for word in sentence]

        labels = -torch.ones(len(classes), dtype=torch.int64)

        for i, c in enumerate(classes):
            if c in self._class_to_idx:
                labels[i] = self._class_to_idx[c]

        return words, labels

    def _load_file_data(self, file):
        root = ET.parse(os.path.join("data", "word-sense-disambiguation", "semcor", file)).getroot()

        words = list(root.findall("word"))

        sentences = [[]]

        for word in words:
            if word.attrib["break_level"] in ["SENTENCE_BREAK", "PARAGRAPH_BREAK"]:
                sentences.append([word])
            else:
                sentences[-1].append(word) 
        
        return sentences

word_sense_files = ['br-a01.xml', 'br-a02.xml', 'br-a03.xml', 'br-a04.xml', 'br-a05.xml', 'br-a06.xml', 'br-a07.xml', 'br-a08.xml', 'br-a09.xml', 'br-a10.xml', 'br-a11.xml', 'br-a12.xml', 'br-a13.xml', 'br-a14.xml', 'br-a15.xml', 'br-a16.xml', 'br-a17.xml', 'br-a18.xml', 'br-a19.xml', 'br-a20.xml', 'br-a21.xml', 'br-a22.xml', 'br-a23.xml', 'br-a24.xml', 'br-a25.xml', 'br-a26.xml', 'br-a27.xml', 'br-a28.xml', 'br-a29.xml', 'br-a30.xml', 'br-a31.xml', 'br-a32.xml', 'br-a33.xml', 'br-a34.xml', 'br-a35.xml', 'br-a36.xml', 'br-a37.xml', 'br-a38.xml', 'br-a39.xml', 'br-a40.xml', 'br-a41.xml', 'br-a42.xml', 'br-a43.xml', 'br-a44.xml', 'br-b01.xml', 'br-b02.xml', 'br-b03.xml', 'br-b04.xml', 'br-b05.xml', 'br-b06.xml', 'br-b07.xml', 'br-b08.xml', 'br-b09.xml', 'br-b10.xml', 'br-b11.xml', 'br-b12.xml', 'br-b13.xml', 'br-b14.xml', 'br-b15.xml', 'br-b16.xml', 'br-b17.xml', 'br-b18.xml', 'br-b19.xml', 'br-b20.xml', 'br-b21.xml', 'br-b22.xml', 'br-b23.xml', 'br-b24.xml', 'br-b25.xml', 'br-b26.xml', 'br-b27.xml', 'br-c01.xml', 'br-c02.xml', 'br-c03.xml', 'br-c04.xml', 'br-c05.xml', 'br-c06.xml', 'br-c07.xml', 'br-c08.xml', 'br-c09.xml', 'br-c10.xml', 'br-c11.xml', 'br-c12.xml', 'br-c13.xml', 'br-c14.xml', 'br-c15.xml', 'br-c16.xml', 'br-c17.xml', 'br-d01.xml', 'br-d02.xml', 'br-d03.xml', 'br-d04.xml', 'br-d05.xml', 'br-d06.xml', 'br-d07.xml', 'br-d08.xml', 'br-d09.xml', 'br-d10.xml', 'br-d11.xml', 'br-d12.xml', 'br-d13.xml', 'br-d14.xml', 'br-d15.xml', 'br-d16.xml', 'br-d17.xml', 'br-e01.xml', 'br-e02.xml', 'br-e03.xml', 'br-e04.xml', 'br-e05.xml', 'br-e06.xml', 'br-e07.xml', 'br-e08.xml', 'br-e09.xml', 'br-e10.xml', 'br-e11.xml', 'br-e12.xml', 'br-e13.xml', 'br-e14.xml', 'br-e15.xml', 'br-e16.xml', 'br-e17.xml', 'br-e18.xml', 'br-e19.xml', 'br-e20.xml', 'br-e21.xml', 'br-e22.xml', 'br-e23.xml', 'br-e24.xml', 'br-e25.xml', 'br-e26.xml', 'br-e27.xml', 'br-e28.xml', 'br-e29.xml', 'br-e30.xml', 'br-e31.xml', 'br-f01.xml', 'br-f02.xml', 'br-f03.xml', 'br-f04.xml', 'br-f05.xml', 'br-f06.xml', 'br-f07.xml', 'br-f08.xml', 'br-f09.xml', 'br-f10.xml', 'br-f11.xml', 'br-f12.xml', 'br-f13.xml', 'br-f14.xml', 'br-f15.xml', 'br-f16.xml', 'br-f17.xml', 'br-f18.xml', 'br-f19.xml', 'br-f20.xml', 'br-f21.xml', 'br-f22.xml', 'br-f23.xml', 'br-f24.xml', 'br-f25.xml', 'br-f33.xml', 'br-f43.xml', 'br-f44.xml', 'br-g01.xml', 'br-g02.xml', 'br-g03.xml', 'br-g04.xml', 'br-g05.xml', 'br-g06.xml', 'br-g07.xml', 'br-g08.xml', 'br-g09.xml', 'br-g10.xml', 'br-g11.xml', 'br-g12.xml', 'br-g13.xml', 'br-g14.xml', 'br-g15.xml', 'br-g16.xml', 'br-g17.xml', 'br-g18.xml', 'br-g19.xml', 'br-g20.xml', 'br-g21.xml', 'br-g22.xml', 'br-g23.xml', 'br-g28.xml', 'br-g31.xml', 'br-g39.xml', 'br-g43.xml', 'br-g44.xml', 'br-h01.xml', 'br-h02.xml', 'br-h03.xml', 'br-h04.xml', 'br-h05.xml', 'br-h06.xml', 'br-h07.xml', 'br-h08.xml', 'br-h09.xml', 'br-h10.xml', 'br-h11.xml', 'br-h12.xml', 'br-h13.xml', 'br-h14.xml', 'br-h15.xml', 'br-h16.xml', 'br-h17.xml', 'br-h18.xml', 'br-h21.xml', 'br-h24.xml', 'br-j01.xml', 'br-j02.xml', 'br-j03.xml', 'br-j04.xml', 'br-j05.xml', 'br-j06.xml', 'br-j07.xml', 'br-j08.xml', 'br-j09.xml', 'br-j10.xml', 'br-j11.xml', 'br-j12.xml', 'br-j13.xml', 'br-j14.xml', 'br-j15.xml', 'br-j16.xml', 'br-j17.xml', 'br-j18.xml', 'br-j19.xml', 'br-j20.xml', 'br-j21.xml', 'br-j22.xml', 'br-j23.xml', 'br-j24.xml', 'br-j25.xml', 'br-j26.xml', 'br-j27.xml', 'br-j28.xml', 'br-j29.xml', 'br-j30.xml', 'br-j31.xml', 'br-j32.xml', 'br-j33.xml', 'br-j34.xml', 'br-j35.xml', 'br-j37.xml', 'br-j38.xml', 'br-j41.xml', 'br-j42.xml', 'br-j52.xml', 'br-j53.xml', 'br-j54.xml', 'br-j55.xml', 'br-j56.xml', 'br-j57.xml', 'br-j58.xml', 'br-j59.xml', 'br-j60.xml', 'br-j70.xml', 'br-k01.xml', 'br-k02.xml', 'br-k03.xml', 'br-k04.xml', 'br-k05.xml', 'br-k06.xml', 'br-k07.xml', 'br-k08.xml', 'br-k09.xml', 'br-k10.xml', 'br-k11.xml', 'br-k12.xml', 'br-k13.xml', 'br-k14.xml', 'br-k15.xml', 'br-k16.xml', 'br-k17.xml', 'br-k18.xml', 'br-k19.xml', 'br-k20.xml', 'br-k21.xml', 'br-k22.xml', 'br-k23.xml', 'br-k24.xml', 'br-k25.xml', 'br-k26.xml', 'br-k27.xml', 'br-k28.xml', 'br-k29.xml', 'br-l01.xml', 'br-l02.xml', 'br-l03.xml', 'br-l04.xml', 'br-l05.xml', 'br-l06.xml', 'br-l07.xml', 'br-l08.xml', 'br-l09.xml', 'br-l10.xml', 'br-l11.xml', 'br-l12.xml', 'br-l13.xml', 'br-l14.xml', 'br-l15.xml', 'br-l16.xml', 'br-l17.xml', 'br-l18.xml', 'br-m01.xml', 'br-m02.xml', 'br-m03.xml', 'br-m04.xml', 'br-m05.xml', 'br-m06.xml', 'br-n01.xml', 'br-n02.xml', 'br-n03.xml', 'br-n04.xml', 'br-n05.xml', 'br-n06.xml', 'br-n07.xml', 'br-n08.xml', 'br-n09.xml', 'br-n10.xml', 'br-n11.xml', 'br-n12.xml', 'br-n14.xml', 'br-n15.xml', 'br-n16.xml', 'br-n17.xml', 'br-n20.xml', 'br-p01.xml', 'br-p02.xml', 'br-p03.xml', 'br-p04.xml', 'br-p05.xml', 'br-p06.xml', 'br-p07.xml', 'br-p08.xml', 'br-p09.xml', 'br-p10.xml', 'br-p12.xml', 'br-p24.xml', 'br-r01.xml', 'br-r02.xml', 'br-r03.xml', 'br-r04.xml', 'br-r05.xml', 'br-r06.xml', 'br-r07.xml', 'br-r08.xml', 'br-r09.xml']


class PennDataset(Dataset):
    '''
    Dataset for English Penn Treebank Wall Street Journal POS
    '''
    def __init__(self, set_name="train", sections=[], first_label=True):
        '''
        Initialise POS dataset.
        Args:
            set_name: use standard sections of the dataset, options: train, dev, test, manual
            sections: if set_name is manual, list sections to be used here
            first_label: set to False to get a list of correct labels instead of the first correct label
        '''
        # standard splits (c.f. A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks)
        sections_standard = {
            'train': [s for s in range(19)],
            'dev': [s for s in range(19, 22)],
            'test': [s for s in range(22,25)]
        }
        self.id_to_label = ['#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
        self.label_to_id = {self.id_to_label[i]: i for i in range(len(self.id_to_label))}

        # obtain split from argument
        sections = sections_standard.get(set_name, sections)

        # Collect data
        data = []
        for section in sections:
            path = os.path.join(PENN_TREEBANK_PATH, "{:02d}".format(section))
            files = os.listdir(path)
            for file_name in files:
                file_path = os.path.join(path, file_name)
                with open(file_path, 'r') as f:
                    content = f.read()
                    data.extend([[self.split_pair(pair) for pair in sent] 
                        for sent in [re.sub(r"[\[\]\n]","",s).split() 
                        for s in re.sub("======================================\n","", content).split("\n\n")] 
                        if sent])

        # Store data in right format
        if first_label:
            self._data = [(
                [pair[0] for pair in sent],
                [self.label_to_id[pair[1][0]] for pair in sent]
                ) for sent in data]
        else:
            self._data = [(
                [pair[0] for pair in sent],
                [[self.label_to_id[label] for label in pair[1]] for pair in sent]
                ) for sent in data]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        '''
        Returns a list of words, and a list of labels.
        If first_label is False, the label is represented by a list of correct labels
        '''
        return self._data[idx]

    def split_pair(self, pair):
        '''returns (word, [POS tags])'''
        split = [x[::-1] for x in pair[::-1].split("/",1)]
        return (split[1], split[0].split("|"))




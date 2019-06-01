import pandas as pd
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from csv import DictReader

import eval_copy as eval
import output

'''
For analysis and visualisation

Ideas:
+ WiC ROC curve
+ t-sne of sentence batch under model
'''

def changing_prediction(pred_file_from, pred_file_to, print_top=1000):
    '''
    Finds all sentences that one model predicts correctly and the other doesn't
    Args:
        pred_file_from, pred_file_to: WiC predictions files
        print_top: if larger than 0, prints top N results based on score difference
    Returns:
        to_correct, to_incorrect: predictions that became correct in pred_file_to or 
            became incorrect, sorted by difference in score. Note that different models
            use different thresholds, so score difference is not a fully informative measure
    '''
    wic_dev_data_file = os.path.join('data','wic','dev','dev.data.txt')
    wic_dev_label_file = os.path.join('data','wic','dev','dev.gold.txt')

    # Read dev data
    with open(wic_dev_data_file) as f:
        reader = DictReader(f, fieldnames=["word","nounverb","positions","sent1","sent2"], delimiter='\t')
        data = list(reader)
    # Read dev true labels
    with open(wic_dev_label_file) as f:
        labels = [line[0] for line in f.readlines()]

    # Read model predictions
    predictions = []
    for pred_file in [pred_file_from, pred_file_to]:
        with open(pred_file) as f:
            reader = DictReader(f, fieldnames=["prediction", "score"])
            predictions.append(list(reader))

    # Select sentences
    to_correct = [] # observations where the prediction was incorrect and becomes correct
    to_incorrect = [] # observations where the prediction was correct and becomes incorrect
    for i in range(len(data)):
        frm = predictions[0][i]
        to = predictions[1][i]
        if frm['prediction'] != to['prediction']:
            obs = {
                'pred_from': frm,
                'pred_to': to,
                'data': data[i],
                'label': labels[i],
                'score_diff': float(to['score']) - float(frm['score'])
            }
            if frm['prediction'] == labels[i]:
                to_incorrect.append(obs)
            else:
                to_correct.append(obs)

    # Sort based on largest score difference
    to_correct = sorted(to_correct, key=lambda obs: abs(obs['score_diff']), reverse=True)
    to_incorrect = sorted(to_incorrect, key=lambda obs: abs(obs['score_diff']), reverse=True)

    # Print top 10
    if print_top > 0:
        print("Predictions that became CORRECT:")
        for obs in to_correct[:print_top]:
            print("\t{} sense of '{}' in:\n\t\t{}\n\t\t{}".format(
                'SAME' if obs['label'] == 'T' else 'DIFFERENT',
                obs['data']['word'],
                obs['data']['sent1'],
                obs['data']['sent2']))
        print("Predictions that became INCORRECT:")
        for obs in to_incorrect[:print_top]:
            print("\tWord {} has {} sense in:\n\t\t{}\n\t\t{}".format(
                'SAME' if obs['label'] == 'T' else 'DIFFERENT',
                obs['data']['word'],
                obs['data']['sent1'],
                obs['data']['sent2']))

    return to_correct, to_incorrect







def sent_eval_table(results_file, output_file):
    '''
    Outputs an image displaying SentEval results in a table
    Args:
        results_file
        output_file: html file name
    Returns:
        pandas table
    '''
    # Load results
    results = torch.load(results_file)['senteval']
    
    # Extract relevant scores
    task_names = []
    metric_names = []
    scores = []
    for name in results:
        task_names.append(name)
        metric_names.append('acc')
        scores.append(results[name][0]) # probably task dependent
    
    # Make table
    table = pd.DataFrame({
        'Task': task_names,
        'Metric': metric_names,
        'Score': scores
    })

    # Output to file
    table.to_html(output_file)
    return table


def wic_compare_pos_datasets(file_pairs=[
                                            ("output/pos/evaluation/results.pt", "output/vpos/evaluation/results.pt"),
                                            ("output/pos-snli/evaluation/results.pt", "output/vpos-snli/evaluation/results.pt"),
                                            ("output/pos-vua-snli/evaluation/results.pt", "output/vpos-vua-snli/evaluation/results.pt"),
                                            ("output/vua-pos/evaluation/results.pt", "output/vua-vpos/evaluation/results.pt")
                                        ]):
    '''
    Compares the accuracy between models trained with vua and wsj pos data
    Args:
        file_pairs: list of tuples with file names of the same model with different pos data
    Returns:
        dict with mean and std of the difference
    '''
    acc_pairs = []

    for f1, f2 in file_pairs:
        print(f1,f2)
        r1 = torch.load(f1)['wic']
        r2 = torch.load(f2)['wic']

        for emb in ['pos', 'vua', 'snli', 'average']:
            acc_pairs.append(
                (r1[emb]['test_accuracy']*100, r2[emb]['test_accuracy']*100)
            )
    
    diff = [i-j for i,j in acc_pairs]
    out = {
        'acc_pairs': acc_pairs,
        'diff': diff,
        'diff_avg': np.average(diff),
        'diff_std': np.std(diff)
    }

    return out


def wic_barplot(results_files_dict, output_file):
    '''
    Makes bar plot comparing WiC accuracies per model and embedding layer.
    Args:
        results_files_dict: dictionary with keys model name, values result file path of the model
        output_file: bar plot is written here
    Returns:
        pandas table
    '''
    # Load data from result files
    data = {
        'input': [],
        'pos': [],
        'met': [],
        'nli': [],
        'average': []
    }
    paper_names = {
        'input': 'input',
        'pos': 'pos',
        'vua': 'met',
        'snli': 'nli',
        'average': 'average'
    }
    names = []

    for name, path in results_files_dict.items():
        names.append(name)
        print(path)
        results = torch.load(path)['wic']
        for emb in paper_names.keys():
            if emb in results.keys():
                data[paper_names[emb]].append(results[emb]['test_accuracy']*100)
            else:
                data[paper_names[emb]].append(None)
    
    # Output to file
    frame = pd.DataFrame(data, index=names)
    frame.plot.bar(figsize=(5,4))
    plt.ylim([50,65])

    plt.savefig(output_file,pad_inches=1,bbox_inches='tight')
    return frame


def wic_table(results_files_dict, output_file, include_thresholds=False, include_train_acc=False):
    '''
    Makes html table comparing WiC accuracies.
    Args:
        results_files_dict: dictionary with keys model name, values result file path of the model
        output_file: html table is written here
        include_thresholds: set to True to include best performing threshold
        include_train_acc: set to True to include best training accuracy
    Returns:
        pandas table
    '''
    # Load data from result files
    names = []
    embeddings = []
    test_accs = []
    train_accs = []
    thresholds = []
    for name, path in results_files_dict.items():
        print(path)
        results = torch.load(path)['wic']
        for emb in results.keys():
            names.append(name)
            embeddings.append(emb)
            test_accs.append("{:.1f}%".format(results[emb]['test_accuracy']*100))
            train_accs.append("{:.1f}%".format(results[emb]['train_accuracy']*100))
            thresholds.append("{:.2f}".format(results[emb]['threshold']))
    
    # Make table
    frame = {'Model': names, 'Embedding': embeddings}
    if include_thresholds:
        frame['Threshold'] = thresholds
    if include_train_acc:
        frame['Train acc'] = train_accs
    frame['Test acc'] = test_accs

    # Output to file
    table = pd.DataFrame(frame)
    table.to_html(output_file)
    return table


def _wic_table(results_files_dict, output_file, include_thresholds=False, include_train_acc=False):
    '''
    DEPRECATED because results now distinguish between different embeddings
    Makes html table comparing WiC accuracies.
    Args:
        results_files_dict: dictionary with keys model name, values result file path of the model
        output_file: html table is written here
        include_thresholds: set to True to include best performing threshold
        include_train_acc: set to True to include best training accuracy
    Returns:
        pandas table
    '''
    # Load data from result files
    names = []
    test_accs = []
    train_accs = []
    thresholds = []
    for name, path in results_files_dict.items():
        print(path)
        results = torch.load(path)['wic']
        names.append(name)
        test_accs.append("{:.1f}%".format(results['test_accuracy']*100))
        train_accs.append("{:.1f}%".format(results['train_accuracy']*100))
        thresholds.append("{:.2f}".format(results['threshold']))
    
    # Make table
    frame = {'Model': names}
    if include_thresholds:
        frame['Threshold'] = thresholds
    if include_train_acc:
        frame['Train acc'] = train_accs
    frame['Test acc'] = test_accs

    # Output to file
    table = pd.DataFrame(frame)
    table.to_html(output_file)
    return table


class WicTsne(object):
    '''
    Apply t-SNE (t-Distributed Stochastic Neighbour Embedding) to the word-of-interest in the WiC train dataset
    Show all contextualized representations of one word
    TODO: implement average embedding
    '''
    def __init__(self):
        '''
        Load WiC train dataset
        '''
        # Load dataset
        self.data = eval.WicEvaluator.load_data(os.path.join(eval.PATH_TO_WIC, 'train', 'train.data.txt'))
        self.labels = eval.WicEvaluator.load_labels(os.path.join(eval.PATH_TO_WIC, 'train', 'train.gold.txt'))

    def set_model(self, model, device='cpu'):
        '''
        Set the model if the words need to be embedded
        Args:
            model: pytorch Module or path to model saved by an OutputWriter
            device: device to use for embeddings
        '''
        # Load model if it is a file path
        if isinstance(model, str):
            print('Loading model')
            model = output.OutputWriter.load_model(model, device=device)
        model.to(device)
        self.model = model

    def embed(self, layer=None):
        '''
        Embeds all words in the training WiC set with the model
        Args:
            layer: which layer's embedding to use, if None the model's embed layer is assumed to return only one embedding (baseline)
        '''

        # Embed all words, use only one embedding per word. 
        # ELMo embedding does not return precisely the same embedding every time
        # embeddings: {word -> {sentence -> word_embedding}}, word has form "word_pos" with pos = N,V
        print("Embedding all words")
        self.all_embeddings = [] # list of all word embeddings, used for t-SNE
        with torch.no_grad():
            self.embeddings = {}
            for i, data_point in enumerate(self.data):
                if i % 100 == 0:
                    print("\t{:02.1f}%".format(i/5428*100))
                # Embed the two sentences
                word = data_point['word'] + "_" + data_point['pos']
                sentence_pair = data_point['sentences']
                word_positions = data_point['positions']
                embedding_sentence = self.model.embed_words(sentence_pair)
                # Add embedding if it was not computed before
                if word not in self.embeddings.keys():
                    self.embeddings[word] = {}
                for n in [0,1]:
                    if tuple(sentence_pair[n]) not in self.embeddings[word].keys():
                        if layer is None:
                            self.embeddings[word][tuple(sentence_pair[n])] = embedding_sentence[n, word_positions[n]].cpu()
                            for emb in embedding_sentence[n].cpu():
                                self.all_embeddings.append(emb)
                        else:
                            self.embeddings[word][tuple(sentence_pair[n])] = embedding_sentence[layer][n, word_positions[n]].cpu()
                            for emb in embedding_sentence[layer][n].cpu():
                                self.all_embeddings.append(emb)

    def compute_tsne(self):
        '''
        Computes t-sne coordinates of all embeddings
        '''
        # Apply t-SNE to all embeddings
        embs = np.vstack(self.all_embeddings)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne_embeddings = tsne.fit_transform(embs)
        # Make (ugly) dict to map word embedding -> tsne coordinate
        self.tsne_map = {tuple(float(val) for val in self.all_embeddings[i]): tsne_embeddings[i] for i in range(len(self.all_embeddings))}

    def map_embedding(self, embedding):
        '''
        Maps a word embedding to a t-SNE coordinate
        '''
        return self.tsne_map[tuple(float(val) for val in embedding)]

    def save_embeddings(self, file):
        '''
        Save model's embeddings to a file
        '''
        torch.save(self.embeddings, file)

    def load_embeddings(self, file):
        '''
        Loads a model's embeddings from a file
        '''
        self.embeddings = torch.load(file, map_location='cpu')

    def save_tsne_map(self, file):
        '''
        Saves map from embedding to t-SNE coordinate to a file
        '''
        torch.save(self.tsne_map, file)

    def load_tsne_map(self, load):
        '''
        Loads map from embedding to t-SNE coordinate to a file
        '''
        self.tsne_map = torch.load(file)

    def word_counts(self):
        '''
        Returns the embedded words and in how many sentences they occur
        '''
        return sorted([(word, len(sentences)) for word, sentences in self.embeddings.items()],
            key=lambda item: item[1], reverse=True)

    def set_word(self, word):
        '''
        Sets the word to analyse
        '''
        self.word = word
        # process links
        emb_dict = self.embeddings[word]
        self.sentences = [sent for sent, _ in emb_dict.items()]
        sentence_to_id = {self.sentences[i]: i for i in range(len(self.sentences))}
        self.links = [] # dicts (ids=[x,y], label=T/F)
        for data_point, label in zip(self.data, self.labels):
            # Embed the two sentences
            # Add link if this is the requested word
            if word == data_point['word']+"_"+data_point['pos']:
                sentence_pair = data_point['sentences']
                self.links.append({
                    'ids': [sentence_to_id[tuple(sentence_pair[i])] for i in [0,1]],
                    'label': label
                })
        # Print
        print("Sentences:")
        for i, sent in enumerate(self.sentences):
            print("\t[{}]  {}".format(i, " ".join(sent)))
        print("Links:")
        for i, link in enumerate(self.links):
            print("\t{} - {} ({})".format(*link['ids'], link['label']))

    def compute_tsne_word(self):
        '''
        Computes t-sne coordinates of all embeddings of a word
        '''
        # Apply t-SNE to the word embeddings
        emb_dict = self.embeddings[self.word]
        embs = np.vstack([embed.detach() for _, embed in emb_dict.items()])
        self.tsne_embeddings = np.vstack([self.map_embedding(emb) for emb in embs])

        # Print coordinates
        print("t-SNE coordinates:")
        for i, embed in enumerate(self.tsne_embeddings):
            print("\t[{}]  ({:.4f}, {:.4f})\t{}".format(i,embed[0], embed[1], " ".join(self.sentences[i])))

    def plot_tsne(self, ids=None, out_file=None):
        '''
        Plot t-SNE for words with the given indices, or all words if ids=None
        If out_file is not None, the figure will be saved to that location
        '''
        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        all_coors = []
        for link in self.links:
            index = link['ids']
            if ids is not None and (index[0] not in ids or index[1] not in ids):
                continue
            # Obtain coordinates
            coors = [self.tsne_embeddings[index[i]] for i in [0,1]]
            
            # Plot text
            for i in [0,1]:
                coor = coors[i]
                if list(coor) not in all_coors:
                    all_coors.append(list(coor))
                    plt.text(coor[0], coor[1], " ".join(self.sentences[index[i]]),
                            fontdict={'size': 9})

            # Link
            plt.plot([coors[0][0],coors[1][0]], [coors[0][1],coors[1][1]],
                color='g' if link['label'] else 'r')


        plt.xticks([]), plt.yticks([])

        all_coors = np.vstack(all_coors)
        min_x = min(all_coors[:,0])
        max_x = max(all_coors[:,0])
        del_x = max_x - min_x
        min_y = min(all_coors[:,1])
        max_y = max(all_coors[:,1])
        del_y = max_y - min_y
        plt.xlim((min_x-del_x/10, max_x+del_x/10))
        plt.ylim((min_y-del_y/10, max_y+del_y/10))
        plt.title('t-SNE of sentences containing "{}"'.format(self.word))

        if out_file is not None:
            plt.savefig(out_file)

        plt.show()



def wicTsne(model, word, output_file):
    # Load dataset
    print("Loading WiC train data")
    data = eval.WicEvaluator.load_data(os.path.join(eval.PATH_TO_WIC, 'train', 'train.data.txt'))
    labels = eval.WicEvaluator.load_labels(os.path.join(eval.PATH_TO_WIC, 'train', 'train.gold.txt'))

    # Extract embedded words
    print("Embed words")
    embeddings = [] 
    links = [] # dicts ([word_ids], [sentences], label=T/F)
    for data_point, label in zip(data, labels):
        # Embed the two sentences
        sentence_pair = data_point['sentences']
        word_positions = data_point['positions']
        embedding_sentence = model.embed_words(sentence_pair)
        # Extract the two word embedding
        embeddings += [embedding_sentence[0, word_positions[0]], embedding_sentence[1, word_positions[1]]]
        # Add link if this is the requested word
        if word == data_point['word']:
            links.append({
                'word_ids': [len(embeddings) - 2, len(embeddings) - 1],
                'sentences': sentence_pair,
                'label': label
            })
        if len(embeddings) % 200 == 0:
            print("\t{:.1f}%".format(len(embeddings) / len(data)/2 *100))
    embeddings = np.vstack(embeddings)

    # Apply t-SNE
    print("Applying t-SNE")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Obtaining word coordinates
    for i in range(len(links)):
        indices = links[i]['word_ids']
        links[i]['tsne'] = [embeddings_tsne[indices[0]], embeddings_tsne[indices[1]]]

    # Plot
    print("Making plot")
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    all_coors = []
    for link in links:
        # Obtain coordinates
        index = link['word_ids']
        coors = [embeddings_tsne[index[0]], embeddings_tsne[index[1]]]
        all_coors += coors

        for i in [0,1]:
            plt.text(coors[i][0], coors[i][1], " ".join(link['sentences'][i]),
                    fontdict={'size': 9})

        plt.plot([coors[0][0],coors[1][0]], [coors[0][1],coors[1][1]],
            color='g' if link['label'] else 'r')


    plt.xticks([]), plt.yticks([])

    all_coors = np.vstack(all_coors)
    min_x = min(all_coors[:,0])
    max_x = max(all_coors[:,0])
    del_x = max_x - min_x
    min_y = min(all_coors[:,1])
    max_y = max(all_coors[:,1])
    del_y = max_y - min_y
    plt.xlim((min_x-del_x/10, max_x+del_x/10))
    plt.ylim((min_y-del_y/10, max_y+del_y/10))
    plt.title('t-SNE of sentences containing "{}"'.format(word))

    plt.savefig(output_file)

import pandas as pd
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import eval

'''
Ideas:
+ WiC ROC curve
+ t-sne of sentence batch under model
'''

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
        'vua': [],
        'snli': [],
        'average': []
    }
    names = []

    for name, path in results_files_dict.items():
        names.append(name)
        print(path)
        results = torch.load(path)['wic']
        for emb in data.keys():
            if emb in results.keys():
                data[emb].append(results[emb]['test_accuracy']*100)
            else:
                data[emb].append(None)
    
    # Output to file
    fig = plt.figure(figsize=(10,6))
    frame = pd.DataFrame(data, index=names)
    frame.plot.bar(rot=0)
    plt.ylim([50,65])

    plt.savefig(output_file)
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


def wicTsne(model, word, output_file):
    '''
    Apply t-SNE (t-Distributed Stochastic Neighbour Embedding) to the word-of-interest in the WiC train dataset
    Show all contextualized representations of one word
    TODO: this should probably become something interactive -> Class
    '''
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
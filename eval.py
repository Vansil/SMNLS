'''
Call this file to evaluate a model checkpoint.
Evaluation methods:
- SentEval
- Word-in-Context
'''

import argparse
import torch
import logging
import os
import sys
import numpy as np

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = os.path.join('data','SentEval')
# path to the NLP datasets 
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')
# path to glove embeddings
PATH_TO_VEC = os.path.join('data', 'glove', 'glove.840B.300d.txt')
# path to WiC data
PATH_TO_WIC = os.path.join('data','wic')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
# import senteval
SENTEVAL_FAST = True # Set to false to perform slower SentEval with better results


##################################################################################################
# SentEval
##################################################################################################

def prepare_senteval(params, samples):
    return

def batcher_senteval(params, batch):
    model = params['evaluated_model']
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = model.embed_sentences(batch).cpu().detach().numpy()
    return embeddings


def eval_senteval(model, output_path):
    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    # Set parameters
    if SENTEVAL_FAST:
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5,
                    'evaluated_model': model}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}
    else:
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10,
                    'evaluated_model': model}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                        'tenacity': 5, 'epoch_size': 4}
    
    # Create SE instance
    se = senteval.engine.SE(params, batcher_senteval, prepare_senteval)

    # Determine tasks
    # Tested succesfully: 'CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 
    transfer_tasks = ['SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    
    # Evaluate
    results = se.eval(transfer_tasks)
    return results



##################################################################################################
# Word-in-Context
##################################################################################################

def eval_wic(model, output_path):
    evaluater = WicEvaluator(model, output_path)
    return evaluater.evaluate()

class WicEvaluator():
    '''
    Container for WiC evaluation functions
    '''
    def __init__(self, model, output_path):
        self.model = model
        self.output_path = output_path

    def evaluate(self):

        results = {}

        # Load datasets
        print("Loading datasets and labels")
        data = {
            'train': self.load_data(os.path.join(PATH_TO_WIC, 'train', 'train.data.txt')),
            'dev':   self.load_data(os.path.join(PATH_TO_WIC, 'dev', 'dev.data.txt'))  
        }
        labels = {
            'train': self.load_labels(os.path.join(PATH_TO_WIC, 'train', 'train.gold.txt')),
            'dev':   self.load_labels(os.path.join(PATH_TO_WIC, 'dev', 'dev.gold.txt'))
        }

        # Extract embedded words
        print("Embed words")
        embeddings = {}
        for set_name in ['train', 'dev']:
            print("\t{} set".format(set_name))
            embeddings[set_name] = []
            for data_point in data[set_name]:
                # Embed the two sentences
                sentence_pair = data_point['sentences']
                word_positions = data_point['positions']
                embedding_sentence = self.model.embed_words(sentence_pair)
                # Extract the two word embedding
                embeddings[set_name].append(
                    (embedding_sentence[0, word_positions[0]].detach(), 
                     embedding_sentence[1, word_positions[1]].detach())
                )

        # Evaluate thresholded cosine similarity metric
        # Compute cosine similarity
        print("Evaluating cosine similarity - threshold method")
        cosine_scores = {}
        for set_name in ['train', 'dev']:
            N = len(embeddings[set_name])
            cosine_scores[set_name] = np.zeros(N)
            for i in range(N):
                emb = embeddings[set_name][i]
                score = emb[0] @ emb[1] / emb[0].norm() / emb[1].norm()
                cosine_scores[set_name][i] = score
        # Find best threshold by trying at every 0.02 interval on the training data
        thresholds = np.linspace(0, 1, 51)
        best_threshold = 0
        best_acc = 0
        train_labels = np.array(labels['train'])
        for threshold in thresholds:
            predictions = cosine_scores['train'] > threshold
            accuracy = (predictions & train_labels).mean()
            if accuracy > best_acc:
                best_threshold = threshold
                best_acc = accuracy
        # Evaluate dev data using threshold
        dev_labels = np.array(labels['dev'])
        predictions = cosine_scores['dev'] > best_threshold
        accuracy = (predictions & dev_labels).mean()
        # add performance to results and write predictions to output file
        results['threshold'] = {
            'threshold': best_threshold,
            'accuracy': accuracy
        }
        with open(os.path.join(self.output_path, 'wic_dev_predictions.txt'), 'w') as f:
            for label in predictions:
                f.write('T\n' if label else 'F\n')


        return results
    

    def load_data(self, data_path):
        '''
        Loads the WiC data from the .txt file as list of dicts
        Example data point:
            {'word': 'board',
            'pos': 'N',
            'positions': [2, 2],
            'sentences': [['Room', 'and', 'board', '.'],
                ['He', 'nailed', 'boards', 'across', 'the', 'windows', '.']]}
        '''
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                elements = line[:-1].split("\t")
                data.append({
                    'word': elements[0],
                    'pos': elements[1],
                    'positions': [int(p) for p in elements[2].split('-')],
                    'sentences': [
                        elements[3].split(),
                        elements[4].split()
                    ]
                })
        return data

    def load_labels(self, labels_path):
        '''
        Loads the WiC labels from the .txt file as list
        Indices correspond to data indices
        Labels: T -> True, F -> False
        '''
        labels = []
        with open(labels_path, 'r') as f:
            for line in f:
                labels.append(line[0] == 'T')
        return labels



##################################################################################################
# Main function
##################################################################################################

if __name__ == "__main__":
    eval_methods = {
        "senteval": eval_senteval,
        "wic": eval_wic
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--method", "-m", type=str, default='all', choices=["all"]+list(eval_methods.keys()),
        help="Evaluation method"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, required=True,
        help="Path to pickle file containing output."
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading model")
    model = torch.load(args.checkpoint, map_location=device)
    print("Device: "+device)
    print("Model:" + str(model))
    
    # Perform evaluation
    if args.method == 'all':
        methods = list(eval_methods.keys())
    else:
        methods = [args.method]
    
    results = {}

    for method in methods:
        print("Starting new evaluation: " + method)
        result = eval_methods[method](model, args.output_path)
        results[method] = result

    # Output results
    torch.save(results, args.output_path)

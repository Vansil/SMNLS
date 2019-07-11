'''
Call this file to evaluate a model checkpoint.
Evaluation methods:
- SentEval
- Word-in-Context
'''
import argparse
import torch
import torch.nn as nn
import logging
import os
import sys
import numpy as np
import output
from ruamel import yaml
import itertools
import sklearn
from pdb import set_trace

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = os.path.join('data','SentEval')
# path to the NLP datasets 
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')
# path to glove embeddings
PATH_TO_VEC = os.path.join('data', 'glove', 'glove.840B.300d.txt')
# path to WiC data
PATH_TO_WIC = os.path.join('data','wic')

# SentEval parameter
SENTEVAL_FAST = True # Set to false to perform slower SentEval with better results

class MLP(nn.Module):
    """
    MLP as per Pilehvar: a simple dense network with 100 hidden neurons (ReLU activation), and one output neuron (sigmoid activation)
    """

    # def __init__(self, in_dim, out_dim=1, hidden_dim=100):
    def __init__(self, in_dim, out_dim=2, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            # nn.Sigmoid(),
            nn.Softmax(),
        )

    def forward(self, input):
        return self.net(input)

##################################################################################################
# SentEval
##################################################################################################

def prepare_senteval(params, samples):
    # TODO: make Glove selection file based on samples, if model has GloVe embedding
    return

def batcher_senteval(params, batch):
    # TODO: load GloVe selection file based on samples
    model = params['evaluated_model']
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = model.embed_sentences(batch).cpu().detach().numpy()
    return embeddings


def eval_senteval(model, output_dir, train_tasks):
    # import SentEval
    sys.path.insert(0, PATH_TO_SENTEVAL)
    import senteval
    
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
    transfer_tasks = ['CR', 'SNLI', 'SICKEntailment']
    #                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
    #                  'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                  'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
    #                  'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    
    # Evaluate
    results = se.eval(transfer_tasks)
    return results



##################################################################################################
# Word-in-Context
##################################################################################################

def eval_wic(model, output_dir):
    evaluater = WicEvaluator(model, output_dir)
    return evaluater.evaluate()

def get_sentences(meta):
    # return {idx: get_sentence(meta, idx) for idx in range(2)}
    return [get_sentence(meta, idx) for idx in range(2)]

def get_sentence(meta, idx):
    pos = meta['positions'][idx]
    meta['sentences'][idx][pos] = f"\emph{{{meta['sentences'][idx][pos]}}}"
    return ' '.join(meta['sentences'][idx])

def get_worst(data, confidences, idxs, reverse):
    confidences_ = confidences[idxs].tolist()
    qualitative = np.array(data['dev'])[idxs]
    ranking = sorted(zip(confidences_, qualitative), key=lambda tpl: tpl[0], reverse=reverse)
    # worst = [{'score':score, 'sentences':get_sentences(meta)} for score, meta in ranking]
    worst = [(round(score, 3), get_sentences(meta)) for score, meta in ranking]
    txt = '\n'.join([f'{score} & {sentences[0]} & {sentences[1]} \\\\' for score, sentences in worst])
    return txt

class WicEvaluator():
    '''
    Container for WiC evaluation functions
    '''
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir

        self.positions = {
            "input": 0,
            "pos": 1,
            "vua": 2,
            "snli": 3
        }

    def evaluate(self, write_output=True):
        results = {}

        # Load datasets
        if not os.path.exists(os.path.join(PATH_TO_WIC, 'train_sub')):
            print("Creating subset of WiC train set")
            WicEvaluator.construct_training_set()
        print("Loading datasets and labels")
        set_names = ['train', 'dev']
        data = {
            'train': WicEvaluator.load_data(os.path.join(PATH_TO_WIC, 'train_sub', 'train_sub.data.txt')),
            'dev':   WicEvaluator.load_data(os.path.join(PATH_TO_WIC, 'dev', 'dev.data.txt'))  
        }
        labels = {
            'train': WicEvaluator.load_labels(os.path.join(PATH_TO_WIC, 'train_sub', 'train_sub.gold.txt')),
            'dev':   WicEvaluator.load_labels(os.path.join(PATH_TO_WIC, 'dev', 'dev.gold.txt'))
        }

        test_out = self.model.embed_words([['test']])
        if isinstance(test_out, tuple):
            layers = ['input','pos','vua','snli','average']
        else:
            layers = ['input']
        nTasks = len(layers)

        # Extract embedded words, embeddings is a dict for tasks to a dict for datasets to a list of tuples
        print("Embed words")
        embeddings = {task: {_set: [] for _set in set_names} for task in layers}
        for set_name in set_names:
            print("\t{} set".format(set_name))
            for data_point in data[set_name]:
                # Embed the two sentences
                sentence_pair = data_point['sentences']
                word_positions = data_point['positions']
                embedding_sentence = self.model.embed_words(sentence_pair)

                # Make list of embeddings (input and per hidden layer)
                if not isinstance(embedding_sentence, tuple):
                    embedding_sentence = [embedding_sentence]
                embedding_sentence = list(embedding_sentence)

                # Add average embedding
                if nTasks > 1:
                    embedding_sentence.append(sum(embedding_sentence[1:]) / (len(embedding_sentence)-1) )

                # Extract the two word embedding
                for i in range(len(embedding_sentence)):
                    embeddings[layers[i]][set_name].append(
                            (embedding_sentence[i][0, word_positions[0]], 
                            embedding_sentence[i][1, word_positions[1]])
                        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Evaluating - {args.classifier} method")

        best_threshold = 0
        for task in layers:
            print("\tEmbedding name: {}".format(task))
            best_acc = 0

            if args.classifier == "threshold":
                # Evaluate thresholded cosine similarity metric
                # Compute cosine similarity per embedding
                scores = {}
                for set_name in ['train', 'dev']:
                    N = len(embeddings[task][set_name])
                    scores[set_name] = np.zeros(N)
                    for i in range(N):
                        emb = embeddings[task][set_name][i]
                        score = emb[0] @ emb[1] / emb[0].norm() / emb[1].norm()
                        scores[set_name][i] = score
                # Find best threshold by trying at every 0.02 interval on the training data
                thresholds = np.linspace(0, 1, 51)
                best_threshold = 0
                train_labels = np.array(labels['train'])
                for threshold in thresholds:
                    predictions = scores['train'] > threshold
                    accuracy = (predictions == train_labels).mean()
                    print("Threshold {} -> Train accuracy: {}".format(threshold, accuracy))
                    if accuracy > best_acc:
                        best_threshold = threshold
                        best_acc = accuracy
                # Evaluate dev data using threshold
                print("Best threshold: {}, Train accuracy: {}".format(best_threshold, best_acc))
                confidences = scores['dev']
                predictions = confidences > best_threshold

            elif args.classifier == "mlp":
                in_dim = 2048  # torch.cat(embeddings[task][set_name][i])
                model = MLP(in_dim).to(device)
                # optimizer: Adam
                lr = 1e-3
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # loss: binary crossentropy
                loss = nn.CrossEntropyLoss()
                # batch_size = 32

                for set_name in ['train', 'dev']:
                    X = torch.stack(list(map(torch.cat, embeddings[task][set_name]))).to(device)
                    Y_hat = model.forward(X)

                    if set_name == 'train':
                        Y = torch.tensor(labels[set_name]).to(device).long()
                        output = loss(Y_hat, Y)
                        # output.backward()
                        # *** RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
                    else:
                        confidences = Y_hat        [:, 1].cpu().detach().numpy()
                        predictions = Y_hat.argmax(dim=1).cpu().detach().numpy()

            elif args.classifier == "svm":
                clf = sklearn.svm.SVC(gamma='scale')
                for set_name in ['train', 'dev']:
                    Y = list(map(int, labels[set_name]))
                    X = torch.stack(list(map(torch.cat, embeddings[task][set_name]))).cpu().numpy()
                    if set_name == 'train':
                        clf.fit(X, Y)
                    else:
                        predictions = clf.predict(X)
                        confidences = predictions  # useless for worst ranking

            dev_labels = np.array(labels['dev'])
            accuracy = (predictions == dev_labels).mean()
            print("Dev accuracy: {}".format(accuracy))

            worst_fp = get_worst(data, confidences, ~dev_labels & predictions, True)
            worst_fn = get_worst(data, confidences, ~predictions & dev_labels, False)

            # add performance to results and write predictions to output file
            results[task] = {
                'threshold': best_threshold,
                'test_accuracy': accuracy,
                'train_accuracy': best_acc
            }
            if write_output:
                with open(os.path.join(self.output_dir, 'false_positives_{}.txt'.format(task)), 'w') as f:
                    f.write(worst_fp)
                with open(os.path.join(self.output_dir, 'false_negatives_{}.txt'.format(task)), 'w') as f:
                    f.write(worst_fn)
                with open(os.path.join(self.output_dir, 'wic_dev_predictions_{}.txt'.format(task)), 'w') as f:
                    for label, score in zip(predictions, confidences):
                        f.write("{},{}\n".format('T' if label else 'F', score))

        return results
    
    @classmethod
    def load_data(cls, data_path):
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

    @classmethod
    def load_labels(cls, labels_path):
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

    
    @classmethod
    def construct_training_set(cls):
        '''
        Construct custom WiC training set
        '''
        n_true = 319
        n_false = 319

        os.makedirs(os.path.join(PATH_TO_WIC, 'train_sub'))

        with open(os.path.join(PATH_TO_WIC, 'train', 'train.data.txt'), 'r') as f_data_train,\
                open(os.path.join(PATH_TO_WIC, 'train', 'train.gold.txt'), 'r') as f_label_train,\
                open(os.path.join(PATH_TO_WIC, 'train_sub', 'train_sub.data.txt'), 'w') as f_data_sub,\
                open(os.path.join(PATH_TO_WIC, 'train_sub', 'train_sub.gold.txt'), 'w') as f_label_sub:
            while n_true > 0 or n_false > 0:
                data_line = f_data_train.readline()
                label_line = f_label_train.readline()
                if n_true > 0 and label_line[0] == 'T':
                    n_true -= 1
                    f_data_sub.write(data_line)
                    f_label_sub.write(label_line)
                elif n_false > 0 and label_line[0] == 'F':
                    n_false -= 1
                    f_data_sub.write(data_line)
                    f_label_sub.write(label_line)



##################################################################################################
# Main function
##################################################################################################

if __name__ == "__main__":
    eval_methods = {
        "senteval": eval_senteval,
        "wic": eval_wic
    }

    train_tasks = [
        "input",
        "snli",
        "pos",
        "vua"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--method", "-m", type=str, default='wic', choices=["all"]+list(eval_methods.keys()),
        help="Evaluation method"
    )
    parser.add_argument('--senteval_methods', type=str, 
        default='CR,MR,MPQA,SUBJ,SST2,SST5,TREC,MRPC,SICKEntailment,SICKRelatedness,STSBenchmark,ImageCaptionRetrieval,STS12,STS13,STS14,STS15,STS16,Length,WordContent,Depth,TopConstituents,BigramShift,Tense,SubjNumber,ObjNumber,OddManOut,CoordinationInversion',
        help='SentEval methods to evaluate')
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True,
        help="Directory for output files."
    )
    parser.add_argument(
        "--classifier", type=str, choices=("mlp", "svm", "threshold"), default="threshold",
        help="classifier used to determine whether WiC words are used in the same sense"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="given the stochasticity of the network optimizer, we report average results for n runs (± standard deviation)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: "+device)
    print("Loading model")
    model = output.OutputWriter.load_model(args.checkpoint, device=device)
    model.eval()
    if model.embedding.has_elmo():
        # Run some batches through ELMo to 'warm it up' (https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism)
        model.embedding.elmo.warm_up()
        print("Model:" + str(model))
    
    # Perform evaluation
    if args.method == 'all':
        methods = list(eval_methods.keys())
    else:
        methods = [args.method]
    
    results = {}

    with torch.no_grad():
        for method in methods:
            print("Starting new evaluation: " + method)
            run_results = [eval_methods[method](model, args.output_dir) for run in args.runs]
            result = run_results[0]
            results = {task:{[res[task][k] for res in run_results] for k in result[task]} for task in result}
            results[method] = results

    # Output results
    torch.save(results, os.path.join(args.output_dir, 'results.pt'))

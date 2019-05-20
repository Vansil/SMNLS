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
import output
from ruamel import yaml

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


def eval_senteval(model, output_dir):
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
    meta['sentences'][idx][pos] = f"<br>{meta['sentences'][idx][pos]}</br>"
    return ' '.join(meta['sentences'][idx])

def get_worst(data, cosine_scores, idxs, reverse):
    similarities = cosine_scores['dev'][idxs].tolist()
    qualitative = np.array(data['dev'])[idxs]
    ranking = sorted(zip(similarities, qualitative), reverse=reverse)
    # worst = [{'score':score, 'sentences':get_sentences(meta)} for score, meta in ranking]
    worst = dict([(score, get_sentences(meta)) for score, meta in ranking])
    return worst

class WicEvaluator():
    '''
    Container for WiC evaluation functions
    '''
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir

    def evaluate(self):

        results = {}

        # Load datasets
        if not os.path.exists(os.path.join(PATH_TO_WIC, 'train_sub')):
            print("Creating subset of WiC train set")
            WicEvaluator.construct_training_set()
        print("Loading datasets and labels")
        data = {
            'train': WicEvaluator.load_data(os.path.join(PATH_TO_WIC, 'train_sub', 'train_sub.data.txt')),
            'dev':   WicEvaluator.load_data(os.path.join(PATH_TO_WIC, 'dev', 'dev.data.txt'))  
        }
        labels = {
            'train': WicEvaluator.load_labels(os.path.join(PATH_TO_WIC, 'train_sub', 'train_sub.gold.txt')),
            'dev':   WicEvaluator.load_labels(os.path.join(PATH_TO_WIC, 'dev', 'dev.gold.txt'))
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
                    (embedding_sentence[0, word_positions[0]], 
                     embedding_sentence[1, word_positions[1]])
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
            accuracy = (predictions == train_labels).mean()
            print("Threshold {} -> Train accuracy: {}".format(threshold, accuracy))
            if accuracy > best_acc:
                best_threshold = threshold
                best_acc = accuracy
        # Evaluate dev data using threshold
        print("Best threshold: {}, Train accuracy: {}".format(best_threshold, best_acc))
        dev_labels = np.array(labels['dev'])
        predictions = cosine_scores['dev'] > best_threshold
        accuracy = (predictions == dev_labels).mean()
        print("Dev accuracy: {}".format(accuracy))

        worst = get_worst(data, cosine_scores, ~dev_labels & predictions, False)
        with open(os.path.join(self.output_dir, 'false_positives.txt'), 'w') as f:
            f.write(yaml.dump(worst, default_flow_style=False))

        worst = get_worst(data, cosine_scores, ~predictions & dev_labels, True)
        with open(os.path.join(self.output_dir, 'false_negatives.txt'), 'w') as f:
            f.write(yaml.dump(worst, default_flow_style=False))

        # add performance to results and write predictions to output file
        results = {
            'threshold': best_threshold,
            'test_accuracy': accuracy,
            'train_accuracy': best_acc
        }
        with open(os.path.join(self.output_dir, 'wic_dev_predictions.txt'), 'w') as f:
            for label, score in zip(predictions, cosine_scores['dev']):
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--method", "-m", type=str, default='all', choices=["all"]+list(eval_methods.keys()),
        help="Evaluation method"
    )
    parser.add_argument('--senteval_methods', type=str, 
        default='CR,MR,MPQA,SUBJ,SST2,SST5,TREC,MRPC,SICKEntailment,SICKRelatedness,STSBenchmark,ImageCaptionRetrieval,STS12,STS13,STS14,STS15,STS16,Length,WordContent,Depth,TopConstituents,BigramShift,Tense,SubjNumber,ObjNumber,OddManOut,CoordinationInversion',
        help='SentEval methods to evaluate')
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True,
        help="Directory for output files."
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading model")
    model = output.OutputWriter.load_model(args.checkpoint, device=device)
    model.eval()
    if model.embedding.has_elmo():
        # Run some batches through ELMo to 'warm it up' (https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism)
        model.embedding.elmo.warm_up()
    print("Device: "+device)
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
            result = eval_methods[method](model, args.output_dir)
            results[method] = result

    # Output results
    torch.save(results, os.path.join(args.output_dir, 'results.pt'))

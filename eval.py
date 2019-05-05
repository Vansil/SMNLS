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

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = os.path.join('data','SentEval')
# path to the NLP datasets 
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data', 'downstream')
# path to glove embeddings
PATH_TO_VEC = os.path.join('data', 'glove', 'glove.840B.300d.txt')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
SENTEVAL_FAST = True # Set to false to perform slower SentEval with better results


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
        help="Evaluation method: choose from " + ", ".join(["all"]+list(eval_methods.keys()))
    )
    # parser.add_argument(
    #     "--output_path", "-o", type=str, required=False,
    #     help="Path to whatever output we want."
    # )
    args = parser.parse_args()

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
    
    for method in methods:
        print("Starting new evaluation: " + method)
        eval_methods[method](model)

    # TODO: Output results


##################################################################################################
# SentEval
##################################################################################################

def prepare_senteval(params, samples):
    return

def batcher_senteval(params, batch):
    model = params['evaluated_model']
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = model.embed_sentences(batch).cpu().numpy()
    return embeddings


def eval_senteval(model):
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
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    # transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
    #                   'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
    #                   'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    
    # Evaluate
    results = se.eval(transfer_tasks)
    print(results)



##################################################################################################
# Word-in-Context
##################################################################################################

def eval_wic(model):
    print("WIC")
    pass
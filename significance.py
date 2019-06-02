import math
import numpy as np
from scipy.stats import binom
import pandas as pd
from permutation_test import permutationtest

# https://gist.github.com/kylebgorman/c8b3fb31c1552ecbaafb
def mcnemar_p(b, c):
    """Computes McNemar's test.

    Args:
      b: the number of "wins" for the first condition.
      c: the number of "wins" for the second condition.

    Returns:
      A p-value for McNemar's test.
    """
    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    return 2. * dist.cdf(x)

def gold_df():
    """load in the gold labels for the word in context (WiC) dataset"""
    df = pd.read_csv('data/wic/dev/dev.gold.txt', names=('gold',))
    df['gold'] = df['gold'].map(lambda x: x == 'T')
    return df

def wic_df():
    """load in the question data for the word in context (WiC) dataset"""
    df = pd.read_csv('data/wic/dev/dev.data.txt', sep='\t', names=('target', 'pos', 'numbers', 'context1', 'context2',))
    words = lambda x: x.split(' ')
    df['context1'] = df['context1'].map(words)
    df['context2'] = df['context2'].map(words)
    return df

def model_df(model, stage, gold):
    """load in dev set model results for the word in context (WiC) dataset"""
    df = pd.read_csv(f'output/{model}/evaluation/wic_dev_predictions_{stage}.txt', names=('pred', 'match'))
    df['pred'] = df['pred'].map(lambda x: x == 'T')
    df['correct'] = list(map(lambda a_b: a_b[0] == a_b[1], zip(gold['gold'], df['pred'])))
    return df

def mcnemar_models(a, b):
    """get mcnemar's test p-value for two WiC model results: are they from the same distribution?"""
    corrects = list(zip(a, b))
    filtered = list(filter(lambda a_b: a_b[0] != a_b[1], corrects))
    total = len(filtered)
    a_wins = len(list(filter(lambda a_b: a_b[0], filtered)))
    b_wins = total - a_wins
    p = mcnemar_p(a_wins, b_wins)
    return p

def gen_combs(models, gold, fn, ratio=1.0):
    """"generate p-values for each combination. params:
    - models: a list of model names
    - fn: a function to calculate p-values
    returns: a generator of {a, b, p}
    """
    col_name = lambda k: f'{k}:{models[k]}'
    corrects = {k:model_df(k, v, gold)['correct'].map(int) for k, v in models.items()}
    size = corrects[list(models.keys())[0]].size
    idxs = np.random.permutation(size)
    idxs = idxs[:math.floor(ratio*len(idxs))]
    corrects = {k:df.iloc[idxs] for k,df in corrects.items()}

    for a in models:
        for b in models:
            print(f'{a} - {b}')
            p = np.nan if a == b else \
                fn(corrects[a], corrects[b])
                # np.inf if (corrects[a] == corrects[b]).all() else 
            yield {'a':col_name(a), 'b':col_name(b), 'p':p}

def significance_pivot(models, gold, fn, file, ratio=1.0):
    """generate and save to html a pivot of p-values"""
    rows = gen_combs(models, gold, fn, ratio)
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='a', columns='b', values='p')
    pivot.to_html(file)

if __name__ == "__main__":
    gold = gold_df()
    # wic = wic_df()
    models = {'baseline_elmo2': 'input', 'baseline_elmo012': 'input', 'vua-snli': 'vua', 'empty_jmt': 'pos'}

    print('mcnemar')
    significance_pivot(models, gold, mcnemar_models,  'results/mcnemar.html', 1.0)
    print('permutation')
    significance_pivot(models, gold, permutationtest, 'results/fishers_permutation.html', 0.01)

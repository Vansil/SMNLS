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

def model_df(model, gold):
    """load in dev set model results for the word in context (WiC) dataset"""
    df = pd.read_csv(f'output/{model}/evaluation/wic_dev_predictions.txt', names=('pred', 'match'))
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

def gen_combs(models, gold, fn):
    """"generate p-values for each combination. params:
    - models: a list of model names
    - fn: a function to calculate p-values
    returns: a generator of {a, b, p}
    """
    for a in models:
        for b in models:
            corrects = [model_df(k, gold)['correct'].map(int) for k in (a, b)]
            p = fn(*corrects) if a != b else np.nan
            yield {'a':a, 'b':b, 'p':p}

def significance_pivot(models, gold, fn, file):
    """generate and save to html a pivot of p-values"""
    rows = gen_combs(models, gold, fn)
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='a', columns='b', values='p')
    pivot.to_html(file)

if __name__ == "__main__":
    gold = gold_df()
    # wic = wic_df()
    models = ['baseline_elmo0', 'baseline_elmo1', 'baseline_elmo2']

    significance_pivot(models, gold, mcnemar_models,  'results/mcnemar.html')
    significance_pivot(models, gold, permutationtest, 'results/fishers_permutation.html')
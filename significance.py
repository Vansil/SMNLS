from scipy.stats import binom
import pandas as pd

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
    df = pd.read_csv('data/wic/dev/dev.gold.txt', names=('gold',))
    df['gold'] = df['gold'].map(lambda x: x == 'T')
    return df

def wic_df():
    df = pd.read_csv('data/wic/dev/dev.data.txt', sep='\t', names=('target', 'pos', 'numbers', 'context1', 'context2',))
    words = lambda x: x.split(' ')
    df['context1'] = df['context1'].map(words)
    df['context2'] = df['context2'].map(words)
    return df

def model_df(model, gold):
    df = pd.read_csv(f'output/{model}/evaluation/wic_dev_predictions.txt', names=('pred', 'match'))
    df['pred'] = df['pred'].map(lambda x: x == 'T')
    df['correct'] = list(map(lambda a_b: a_b[0] == a_b[1], zip(gold['gold'], df['pred'])))
    return df

def mcnemar_models(name_a, name_b, gold):
    df_a = model_df(name_a, gold)
    df_b = model_df(name_b, gold)
    corrects = list(zip(df_a['correct'], df_b['correct']))
    filtered = list(filter(lambda a_b: a_b[0] != a_b[1], corrects))
    total = len(filtered)
    a_wins = len(list(filter(lambda a_b: a_b[0], filtered)))
    b_wins = total - a_wins
    p = mcnemar_p(a_wins, b_wins)
    return p

def gen_combs(models, gold):
    for a in models:
        for b in models:
            p = mcnemar_models(a, b, gold)
            yield (a, b, p)

gold = gold_df()
wic = wic_df()
models = ['baseline_elmo0', 'baseline_elmo1', 'baseline_elmo2']
df = pd.DataFrame(gen_combs(models, gold), columns=('a', 'b', 'p'))
pivot = df.pivot(index='a', columns='b', values='p')
pivot.to_html('results/significance.html')

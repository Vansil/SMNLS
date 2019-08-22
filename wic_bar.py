"""display our Word in Context (WiC) results in a bar plot"""

import visual

baselines = ('baseline_elmo0', 'baseline_elmo1', 'baseline_elmo2', 'baseline_elmo012')
base_stages = ('input',)
elmo_stages = ('snli', 'vpos', 'vpos-snli', 'vpos-vua-snli', 'vua', 'vua-snli', 'vua-vpos')
bert_stages = ('snli', 'pos', 'pos-snli', 'pos-vua-snli', 'vua', 'vua-snli', 'pos-vua')
pairs = [(model, stages) for model in baselines for stages in base_stages] + \
        [(model, stages) for model in ('elmo-2', 'elmo-3') for stages in elmo_stages] + \
        [(f'bert-{size}-{i}', stages) for size in ('base', 'large') for i in ('0', '1', '2', '3', '4') for stages in bert_stages]
results_files_dict = {f'{model}_{stages}': f"output/{model}/{stages}/evaluation/results.pt" for model, stages in pairs}

visual.wic_barplot(results_files_dict, 'results/WicBoxplotForPaper.png')

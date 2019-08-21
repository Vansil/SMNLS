"""display our Word in Context (WiC) results as an HTML table"""

import visual

elmo_stages = ('snli', 'vpos', 'vpos-snli', 'vpos-vua-snli', 'vua', 'vua-snli', 'vua-vpos')
bert_stages = ('snli', 'pos', 'pos-snli', 'pos-vua-snli', 'vua', 'vua-snli', 'pos-vua')
pairs = [(model, stages) for model in ('elmo-2', 'elmo-3') for stages in elmo_stages] + \
         [(f'bert-{size}-{i}', stages) for size in ('base', 'large') for i in ('0', '1', '2', '3', '4') for stages in bert_stages]
results_files_dict = {f'{model}_{stages}': f"output/{model}/{stages}/evaluation/results.pt" for model, stages in pairs}

visual.wic_table(results_files_dict, 'results/WicTableVuapos.html', True, True)

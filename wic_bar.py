"""display our Word in Context (WiC) results in a bar plot"""

import visual
results_files_dict = {
    'elmo0': "output/baseline_elmo0/evaluation/results.pt",
    'elmo1': "output/baseline_elmo1/evaluation/results.pt",
    'elmo2': "output/baseline_elmo2/evaluation/results.pt",
    'elmo012': "output/baseline_elmo012/evaluation/results.pt",
    'random': "output/empty_jmt/evaluation/results.pt",
    'pos': "output/vpos/evaluation/results.pt",
    'met': "output/vua/evaluation/results.pt",
    'nli': "output/snli/evaluation/results.pt",
    'pos-met': "output/vua-vpos/evaluation/results.pt",
    'pos-nli': "output/vpos-snli/evaluation/results.pt",
    'met-nli': "output/vua-snli/evaluation/results.pt",
    'pos-met-nli': "output/vpos-vua-snli/evaluation/results.pt",
}
visual.wic_barplot(results_files_dict, 'results/WicBarplotForPaper.png')
